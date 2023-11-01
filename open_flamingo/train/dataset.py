import functools
import logging
import math
import random
import sys
import transformers

import torch
import webdataset as wds
from PIL import Image

from groundingdino.demo.caption_grounder import caption_grounder
from groundingdino.demo.inference_on_laion import add_loc_to_text
from groundingdino.demo.inference_on_laion import nms_without_score
from torch.utils.data import IterableDataset
from data_utils import (
    SharedEpoch,
    ResampledShards2,
    tarfile_to_samples_nothrow,
    _SAMPLE_SHUFFLE_SIZE,
    _SAMPLE_SHUFFLE_INITIAL,
    log_and_continue,
    DataInfo,
    filter_no_caption_or_no_image
)


Image.MAX_IMAGE_PIXELS = 1000000000
LAION2B_NUM_SAMPLE = 1500000000
VQAV2_TRAIN_NUM_SAMPLE = 1828467
VG_RELATION_BBOX_SIZE = 600

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

# DEBUG
log_and_continue = None
# DEBUG


class ConcatDataset(IterableDataset):
    def __init__(
            self, dataset, max_length,
            delimiter_id, pad_id=None, media_id=None, endofmedia_id=None,
            image_embedding_size=-2, single=False, box_id=None, visual_id=None,
    ):
        self.dataset = dataset
        self.max_length = max_length
        self.delimiter_id = torch.ones(1, 1).long() * delimiter_id
        if pad_id is not None:
            self.pad_id = int(pad_id)
        if media_id is not None:
            self.media_id = torch.ones(1, 1).long() * int(media_id)
        if endofmedia_id is not None:
            self.endofmedia_id = torch.ones(1, 1).long() * int(endofmedia_id)
        if image_embedding_size > 0:
            logging.info(f"image_embedding_size: {image_embedding_size}")
        self.image_embedding_size = image_embedding_size + 2
        self.single = single
        self.box_id = box_id
        self.visual_id = visual_id

    def __iter__(self):
        while True:
            input_ids_list = []
            attention_mask_list = []
            image_list = []
            image_start_index_list = []
            added_bbox_list = []
            cnt = 0
            while cnt < self.max_length:
                sample = next(self.dataset)
                if len(sample) >= 4:
                    image = sample[0].unsqueeze(0)
                    input_ids = sample[1]
                    attention_mask = sample[2]
                    added_bbox = sample[3]
                    image_list.append(image)
                    added_bbox_list.append(added_bbox)
                else:
                    sample = sample[0]
                    input_ids = sample[0]
                    attention_mask = sample[1]
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                cnt += input_ids.shape[-1]
                if self.single:
                    break
            input_ids = torch.cat(input_ids_list, dim=-1)[0]
            attention_mask = torch.cat(attention_mask_list, dim=-1)[0]
            if not self.single:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            # TODO: fix visual number not match
            if len(image_list) != 0:
                images = torch.cat(image_list, dim=0)
                image_begin = (input_ids == self.media_id[0, 0]).nonzero().view(-1)
                image_end = (input_ids == self.endofmedia_id[0, 0]).nonzero().view(-1)
                if len(image_begin) != len(image_end):
                    assert len(image_begin) == len(image_end) + 1
                    input_ids[image_begin[-1]:] = self.pad_id
                    attention_mask[image_begin[-1]:] = 0
                    image_begin = image_begin[:-1]
                eos_token_num = len((input_ids == self.delimiter_id[0, 0]).nonzero().view(-1))
                if eos_token_num != len(image_begin) + 1:
                    input_ids[image_begin[-1]:] = self.pad_id
                    attention_mask[image_begin[-1]:] = 0
                    image_begin = image_begin[:-1]
                    image_end = image_end[:-1]
                images = images[:len(image_end)]
                added_bbox_list = added_bbox_list[:len(image_end)]
                image_start_index_list = (image_begin + 1).tolist()
                expand_list = added_bbox_list[0]
                for x in added_bbox_list[1:]:
                    expand_list.extend(x)
                yield images, len(images), image_start_index_list, input_ids, attention_mask, expand_list
            else:
                yield input_ids, attention_mask


def preprocess_image(sample, image_processor):
    image = image_processor(sample)
    if isinstance(image, transformers.image_processing_utils.BatchFeature):
        image = torch.tensor(image["pixel_values"][0])
    return image


def preprocess_text(sample, tokenizer, max_length, single=False):
    if "llama" in tokenizer.__class__.__name__.lower():
        sample = sample.strip()
    else:
        sample = tokenizer.bos_token + sample.strip()
    if not single:
        text = tokenizer(sample, return_tensors="pt", max_length=max_length, truncation=True)
    else:
        text = tokenizer(sample, return_tensors="pt", max_length=max_length, truncation=True, padding='max_length')
    return text["input_ids"], text["attention_mask"]


def preprocess_encoded_text(sample, tokenizer, max_length):
    sample = sample.decode("utf-8")
    return preprocess_text(sample, tokenizer, max_length=max_length)


def _merge_bbox_previsual(added_bbox_list):
    bbox_list = []
    for bboxes in added_bbox_list:
        x1 = bboxes[:, 0].min()
        y1 = bboxes[:, 1].min()
        x2 = bboxes[:, 2].max()
        y2 = bboxes[:, 3].max()
        bbox_list.append(torch.tensor([x1, y1, x2, y2], device=bboxes.device, dtype=bboxes.dtype).unsqueeze(0))
    return bbox_list


def _find_idx(text, subtext):
    loc = 0
    locs = []
    while text.find(subtext, loc) != -1:
        loc = text.find(subtext, loc)
        locs.append(loc)
        loc += len(subtext)
    return locs


def preprocess_ground_caption(sample, image_processor, tokenizer, image_embedding_size, generator, max_length=None, args=None):
    assert max_length is not None
    image, caption, logits_filt, boxes_filt = sample
    image = preprocess_image(image, image_processor=image_processor)
    added_bbox = []
    boxes_filt, pred_phrases = generator.postprocess(
        logits_filt, boxes_filt, generator.ground_model,
        caption, generator.text_threshold, generator.box_threshold,
        with_logits=True,
    )
    caption, added_bbox = add_loc_to_text(
        boxes_filt, pred_phrases, caption,
        expand=True,
    )
    visual_loc = []
    visual_token = "<|#visual#|>"
    previsual_token = "<|#previsual#|>"
    box_token = "<|#box#|>"
    prebox_token = "<|#prebox#|>"
    end_token = "<|#endofobject#|>"
    object_token = "<|#object#|>"
    end_of_attr_token = "<|#endofattr#|>"
    preend_of_attr_token = "<|#preendofattr#|>"
    visual_loc = _find_idx(caption, visual_token)
    if len(visual_loc) != len(added_bbox):
        logging.warning(f"visual_loc: {visual_loc}")
        logging.warning(f"added_bbox: {added_bbox}")
    assert len(visual_loc) == len(added_bbox)
    delta = 0
    for i, (loc, boxes) in enumerate(zip(visual_loc, added_bbox)):
        loc += delta
        boxes = nms_without_score(boxes)
        added_bbox[i] = boxes
        added_tokens = end_token + visual_token + box_token * len(boxes) + end_of_attr_token
        caption = caption[:loc] + added_tokens + caption[len(visual_token) + loc:]
        delta += len(added_tokens) - len(visual_token)

    merge_added_bbox = _merge_bbox_previsual(added_bbox)
    # step 1: move <|#object#|> before the space char
    while caption.find(f" {object_token}") != -1:
        caption = caption.replace(f" {object_token}", f"{object_token} ")
    # step 2: add <|#previsual#|> after <|#object#|> for 75% except the first object
    i = 0
    II = -1
    flag = True
    delete_visual_prob = 0.75
    while i < len(caption):
        if caption[i: i + len(object_token)] == object_token:
            II += 1
            if not flag and random.random() < delete_visual_prob:
                # delete visual and add previsual
                visual_start_idx = caption.find(end_token, i + 1) + len(end_token)
                visual_end_idx = caption.find(end_of_attr_token, visual_start_idx + 1) + len(end_of_attr_token)
                caption = caption[:visual_start_idx] + caption[visual_end_idx:]
                caption = caption[:i + len(object_token)] + previsual_token + prebox_token + preend_of_attr_token + caption[i + len(object_token):]
                added_bbox[II] = merge_added_bbox[II]
        i += 1
        flag = False
    caption = caption.replace(preend_of_attr_token, object_token).replace(end_of_attr_token, end_token)
    caption = f"<|#image#|>{tokenizer.pad_token*image_embedding_size}<|#endofimage#|>" + caption
    if "llama" in tokenizer.__class__.__name__.lower():
        caption = caption.replace(f"{object_token} ", f"{object_token}").replace(f"{end_token} ", f"{end_token}")

    input_ids, attention_mask = preprocess_text(caption, tokenizer, max_length=max_length)
    return image, input_ids, attention_mask, added_bbox


def get_pile_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    input_shards = args.pile_shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)
    assert resampled, "turn on dataset_resampled to allow infinite stream of samples"

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_text_fn = functools.partial(preprocess_encoded_text, tokenizer=tokenizer, max_length=args.max_length)
    pipeline = [
        ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch),
        tarfile_to_samples_nothrow,
        wds.shuffle(
            bufsize=_SAMPLE_SHUFFLE_SIZE,
            initial=_SAMPLE_SHUFFLE_INITIAL,
        ),
        wds.to_tuple("txt", handler=log_and_continue),
        wds.map_tuple(
            preprocess_text_fn, handler=log_and_continue
        ),
    ]
    # with_epoch(sys.maxsize) will give us an infinite sample stream
    dataset = wds.DataPipeline(*pipeline).with_epoch(sys.maxsize)
    delimiter_id = tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"][-1]
    dataset = ConcatDataset(iter(dataset), max_length=args.max_length, delimiter_id=delimiter_id)

    def text_collate_fn(items):
        input_ids = torch.cat([x[0].unsqueeze(0) for x in items], dim=0)
        attention_mask = torch.cat([x[1].unsqueeze(0) for x in items], dim=0)
        return input_ids, attention_mask

    dataloader = wds.WebLoader(
        dataset,
        batch_size=args.batch_size_pile,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=False,
        collate_fn=text_collate_fn,
    )
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


# modify /gpfs/u/home/LMCG/LMCGljnn/scratch/miniconda3-ppc64le/envs/unified/lib/python3.9/site-packages/webdataset/filters.py, line 433
# combine_tensors=True to combine_tensors=False
def get_ground_laion_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    input_shards = args.caption_shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)
    assert resampled, "turn on dataset_resampled to allow infinite stream of samples"
    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    generator = caption_grounder(
        config_file="GroundingDINO_SwinT_OGC.py",
        checkpoint_path=None,
        cpu_only=True,
        # box_threshold=0.5, text_threshold=0.3,
    )
    preprocess_ground_caption_fn = functools.partial(
        preprocess_ground_caption, image_processor=image_processor, tokenizer=tokenizer,
        image_embedding_size=args.vis_embed_size, generator=generator,
        max_length=args.max_length, args=args,
    )
    pipeline = [
        ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch),
        tarfile_to_samples_nothrow,
        wds.shuffle(
            bufsize=_SAMPLE_SHUFFLE_SIZE,
            initial=_SAMPLE_SHUFFLE_INITIAL,
        ),
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", partial=True, handler=log_and_continue),
        wds.to_tuple("jpg;png;jpeg", "txt", "logits.pyd", "boxes.pyd", handler=log_and_continue),
        wds.map(
            preprocess_ground_caption_fn, handler=log_and_continue
        ),
    ]

    dataset = wds.DataPipeline(*pipeline).with_epoch(sys.maxsize)
    media_token_id = tokenizer("<|#image#|>", add_special_tokens=False)["input_ids"][-1]
    delimiter_id = tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"][-1]
    endofmedia_token_id = tokenizer("<|#endofimage#|>", add_special_tokens=False)["input_ids"][-1]
    box_id = tokenizer("<|#box#|>", add_special_tokens=False)["input_ids"][-1]
    visual_id = tokenizer("<|#visual#|>", add_special_tokens=False)["input_ids"][-1]
    dataset = ConcatDataset(
        iter(dataset), max_length=args.max_length,
        delimiter_id=delimiter_id,
        pad_id=tokenizer.pad_token_id,
        media_id=media_token_id,
        endofmedia_id=endofmedia_token_id,
        box_id=box_id,
        visual_id=visual_id,
        image_embedding_size=args.vis_embed_size,
        single=False,
    )

    def image_collate_fn(items):
        images = torch.cat([x[0] for x in items], dim=0)
        image_nums = [x[1] for x in items]
        image_start_index_list = [x[2] for x in items]
        input_ids = torch.cat([x[3].unsqueeze(0) for x in items], dim=0)
        attention_mask = torch.cat([x[4].unsqueeze(0) for x in items], dim=0)
        added_bbox_list = [x[5] for x in items]
        expand_list = added_bbox_list[0]
        for x in added_bbox_list[1:]:
            expand_list.extend(x)
        return images, image_nums, image_start_index_list, input_ids, attention_mask, expand_list

    dataloader = wds.WebLoader(
        dataset,
        batch_size=args.batch_size_caption,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=False,
        collate_fn=image_collate_fn,
    )
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_caption * args.world_size
    num_batches = round_fn(LAION2B_NUM_SAMPLE / global_batch_size)
    dataloader.num_batches = num_batches
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_dataset_fn(dataset_type):
    if dataset_type == "pile":
        return get_pile_dataset
    elif dataset_type == "ground_image_text":
        return get_ground_laion_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, image_processor, tokenizer, dataset_type, epoch=0):
    return get_dataset_fn(dataset_type)(
        args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer
    )

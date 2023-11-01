import argparse
import json
from math import ceil
import os
import random
import uuid
from collections import defaultdict
from typing import Callable
import time
import cv2
import webdataset as wds
import transformers
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from sklearn.metrics import recall_score, average_precision_score

import more_itertools
import numpy as np
import torch
from coco_metric import compute_cider, postprocess_captioning_generation
from eval_datasets import VQADataset
from tqdm import tqdm
from collections import Counter
from vqa_metric import compute_vqa_accuracy, compute_gqa_accuracy
from open_flamingo.src.factory import create_model_and_transforms
from PIL import Image
from io import BytesIO
import base64
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
import string
from open_flamingo.eval.task.reg import evaluate_reg
from open_flamingo.eval.task.gqa import GQADataset
from open_flamingo.eval.task.vl_checklist import evaluate_vlc
from open_flamingo.eval.task.crepe import evaluate_crepe
from open_flamingo.eval.task.caption import evaluate_coco_flickr
from open_flamingo.eval.task.utils import is_correct, get_iou
from open_flamingo.eval.task.cola import evaluate_cola
from open_flamingo.eval.task.gqa import evaluate_gqa

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

parser = argparse.ArgumentParser()
parser.add_argument("--lm_path", type=str, default="facebook/opt-1.3b")
parser.add_argument("--lm_tokenizer_path", type=str, default="facebook/opt-30b")
parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    default=[0],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples", type=int, default=5000, help="Number of samples to evaluate on"
)

parser.add_argument("--batch_size", type=int, default=8)

# Per-dataset evaluation flags
parser.add_argument(
    "--eval_coco",
    action="store_true",
    default=False,
    help="Whether to evaluate on COCO.",
)
parser.add_argument(
    "--eval_vqav2",
    action="store_true",
    default=False,
    help="Whether to evaluate on VQAV2.",
)
parser.add_argument(
    "--eval_ok_vqa",
    action="store_true",
    default=False,
    help="Whether to evaluate on OK-VQA.",
)
parser.add_argument(
    "--eval_imagenet",
    action="store_true",
    default=False,
    help="Whether to evaluate on ImageNet.",
)

parser.add_argument(
    "--eval_flickr30",
    action="store_true",
    default=False,
    help="Whether to evaluate on Flickr30.",
)

parser.add_argument(
    "--eval_refcoco",
    action="store_true",
    default=False,
    help="Whether to evaluate on RefCOCO.",
)

# Dataset arguments

## Flickr30 Dataset
parser.add_argument(
    "--flickr_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default=None,
)
parser.add_argument(
    "--flickr_annotations_json_path",
    type=str,
    help="Path to the dataset_flickr30k_coco_style.json file.",
    default=None,
)

## COCO Dataset
parser.add_argument(
    "--coco_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default=None,
)
parser.add_argument(
    "--coco_annotations_json_path",
    type=str,
    default=None,
)

## VQAV2 Dataset
parser.add_argument(
    "--vqav2_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_annotations_json_path",
    type=str,
    default=None,
)

## OK-VQA Dataset
parser.add_argument(
    "--ok_vqa_image_dir_path",
    type=str,
    help="Path to the vqav2/train2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_train2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_train2014_annotations.json file.",
    default=None,
)

## Imagenet dataset
parser.add_argument("--imagenet_root", type=str, default="/tmp")

## RefCOCO dataset
parser.add_argument("--refcoco_tsvfile", type=str, default=None)

parser.add_argument(
    "--location_token_num",
    default=1000,
    type=int,
)
# distributed training
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--horovod",
    default=False,
    action="store_true",
    help="Use horovod for distributed training.",
)
parser.add_argument(
    "--no-set-device-rank",
    default=False,
    action="store_true",
    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
)
parser.add_argument(
    "--dist",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--lora",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--lora_r",
    default=16,
    type=int,
    required=False,
)
parser.add_argument(
    "--legacy",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--special",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--id",
    default=0,
    type=int,
    required=False,
)

parser.add_argument(
    "--eval_gqa",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--use_sam",
    default=None,
    type=str,
    required=False,
)
parser.add_argument(
    "--add_visual_token",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--use_format_v2",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--eval_aro",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--eval_pisc",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--eval_reg",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--eval_vlc",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--eval_crepe",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--eval_cola",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--level",
    default=4,
    type=int,
)
parser.add_argument(
    "--type",
    default="swap",
    type=str,
)
parser.add_argument(
    "--choose_left_right",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--eval_exp",
    default=False,
    action="store_true",
)


def preprocess_image(sample, image_processor):
    image = image_processor(sample)
    if isinstance(image, transformers.image_processing_utils.BatchFeature):
        image = torch.tensor(image["pixel_values"][0])
    return image


class OKVQAPostProcess():
    def __init__(self):
        self._lemmatizer = None

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer


def main():
    args = parser.parse_args()
    if args.dist:
        args.local_rank, args.rank, args.world_size = world_info_from_env()
        print(f"local_rank: {args.local_rank} rank: {args.rank} world_size: {args.world_size}")
        device_id = init_distributed_device(args)
    else:
        args.rank = 0
        args.world_size = 1
        print(f"rank: {args.rank} world_size: {args.world_size}")

    if "sam" in args.checkpoint_path:
        args.use_sam = "vit_l"

    args.add_visual_token = True
    if "lora" in args.checkpoint_path:
        args.lora = True

    args.add_pe = False
    args.add_box = True
    args.relation = False
    args.enhance_data = False
    args.use_format_v2 = True

    import hashlib
    args.id = hashlib.sha224(args.checkpoint_path.encode()).hexdigest()

    # load model
    flamingo, image_processor, tokenizer, vis_embed_size = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.lm_tokenizer_path,
        location_token_num=args.location_token_num,
        lora=args.lora,
        lora_r=16,
        use_sam=args.use_sam,
        add_visual_token=args.add_visual_token,
        use_format_v2=args.use_format_v2,
        add_box=args.add_box,
        add_pe=args.add_pe,
        add_relation=args.relation,
        enhance_data=args.enhance_data,
        checkpoint_activations=False,
    )
    flamingo.use_format_v2 = args.use_format_v2
    if args.special:
        flamingo.special = True
    else:
        flamingo.special = False
    if args.legacy:
        flamingo.legacy = True
        print("use legacy evaluation")
    flamingo.step_num = int(args.checkpoint_path.split("/")[-1].split(".")[0].split("_")[-1])
    flamingo.expr_name = args.checkpoint_path.split("/")[-2]
    if args.rank == 0:
        print("legacy", True if hasattr(flamingo, "legacy") else False)
        print("step:", flamingo.step_num)
        print("expr:", flamingo.expr_name)
        print("use format v2:", flamingo.use_format_v2)
        print(args)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model_state_dict = {}
    for key in checkpoint["model_state_dict"].keys():
        model_state_dict[key.replace("module.", "")] = \
            checkpoint["model_state_dict"][key]
    _save_path = args.id+"temp.pth"
    torch.save(model_state_dict, _save_path)
    del model_state_dict
    flamingo = load_checkpoint_and_dispatch(
        flamingo, checkpoint=_save_path, device_map="auto",
        no_split_module_classes=['FlamingoLayer'],
    )
    os.remove(_save_path)
    results = defaultdict(list)
    if args.eval_coco:
        print("Evaluating on COCO...")
        cider_score = evaluate_coco_flickr(
            model=flamingo,
            tokenizer=tokenizer,
            image_processor=image_processor,
            batch_size=args.batch_size,
            vis_embed_size=vis_embed_size,
            rank=args.rank,
            world_size=args.world_size,
            id=args.id,
        )
        results["coco"].append({"score": cider_score})

    if args.eval_ok_vqa:
        print("Evaluating on OK-VQA...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                ok_vqa_score = evaluate_vqa(
                    model=flamingo,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    batch_size=args.batch_size,
                    image_dir_path=args.ok_vqa_image_dir_path,
                    questions_json_path=args.ok_vqa_questions_json_path,
                    annotations_json_path=args.ok_vqa_annotations_json_path,
                    vqa_dataset="ok_vqa",
                    vis_embed_size=vis_embed_size,
                    rank=args.rank,
                    world_size=args.world_size,
                    id=args.id,
                )
            results["ok_vqa"].append(
                {"shots": shot, "score": ok_vqa_score}
            )

    if args.eval_vqav2:
        print("Evaluating on VQAv2...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vqa_score = evaluate_vqa(
                    model=flamingo,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    batch_size=args.batch_size,
                    image_dir_path=args.vqav2_image_dir_path,
                    questions_json_path=args.vqav2_questions_json_path,
                    annotations_json_path=args.vqav2_annotations_json_path,
                    vqa_dataset="vqa",
                    vis_embed_size=vis_embed_size,
                    rank=args.rank,
                    world_size=args.world_size,
                    id=args.id,
                )
            results["vqav2"].append(
                {"shots": shot, "score": vqa_score}
            )

    if args.eval_gqa:
        print("Evaluating on GQA...")
        gqa_score = evaluate_gqa(
            model=flamingo,
            tokenizer=tokenizer,
            image_processor=image_processor,
            batch_size=args.batch_size,
            vis_embed_size=vis_embed_size,
            rank=args.rank,
            world_size=args.world_size,
            id=args.id,
        )
        results["gqa"].append(
            {"score": gqa_score}
        )

    if args.eval_refcoco:
        print("Evaluating on RefCOCO...")
        refcoco_score = evaluate_refcoco(
            model=flamingo,
            tokenizer=tokenizer,
            image_processor=image_processor,
            tsvfile=args.refcoco_tsvfile,
            vis_embed_size=vis_embed_size,
            rank=args.rank,
            world_size=args.world_size,
            id=args.id,
        )
        results["refcoco"].append(
            {"score": refcoco_score}
        )
    if args.eval_aro:
        print("Evaluating on ARO...")
        aro_score = evaluate_aro(
            model=flamingo,
            tokenizer=tokenizer,
            image_processor=image_processor,
            vis_embed_size=vis_embed_size,
            rank=args.rank,
            world_size=args.world_size,
            id=args.id,
            choose_left_right=args.choose_left_right,
        )
        results["aro"].append(
            {"score": aro_score}
        )
    if args.eval_pisc:
        print("Evaluating on ARO...")
        aro_score = evaluate_pisc(
            model=flamingo,
            tokenizer=tokenizer,
            image_processor=image_processor,
            batch_size=args.batch_size,
            device=args.device,
            tsvfile=args.refcoco_tsvfile,
            vis_embed_size=vis_embed_size,
            rank=args.rank,
            world_size=args.world_size,
            id=args.id,
        )
        results["pisc"].append(
            {"score": aro_score}
        )
    if args.eval_reg:
        print("Evaluating on Referring Expression Generation...")
        cider = evaluate_reg(
            model=flamingo,
            tokenizer=tokenizer,
            image_processor=image_processor,
            vis_embed_size=vis_embed_size,
            rank=args.rank,
            world_size=args.world_size,
            id=args.id,
        )
        results["reg"].append(
            {"score": cider}
        )
    if args.eval_vlc:
        print("Evaluating on VL-checklist...")
        vlc_score = evaluate_vlc(
            model=flamingo,
            tokenizer=tokenizer,
            image_processor=image_processor,
            vis_embed_size=vis_embed_size,
            rank=args.rank,
            world_size=args.world_size,
            id=args.id,
        )
        results["vlc"].append(
            {"score": vlc_score}
        )
    if args.eval_crepe:
        print("Evaluating on CREPE...")
        crepe_score = evaluate_crepe(
            model=flamingo,
            tokenizer=tokenizer,
            image_processor=image_processor,
            vis_embed_size=vis_embed_size,
            rank=args.rank,
            world_size=args.world_size,
            id=args.id,
            level=args.level,
            type=args.type,
        )
        results["crepe"].append(
            {"score": crepe_score}
        )
    if args.eval_cola:
        print("Evaluating on COLA...")
        cola_score = evaluate_cola(
            model=flamingo,
            tokenizer=tokenizer,
            image_processor=image_processor,
            vis_embed_size=vis_embed_size,
            rank=args.rank,
            world_size=args.world_size,
            id=args.id,
        )
        results["cola"].append(
            {"score": cola_score}
        )
    if args.eval_exp:
        evaluate_exp(
            model=flamingo,
            tokenizer=tokenizer,
            image_processor=image_processor,
            vis_embed_size=vis_embed_size,
            rank=args.rank,
            world_size=args.world_size,
            id=args.id,
        )


def prepare_batch_images(batch, image_processor):
    batch_images = None
    for b in batch:
        b_image = image_processor(b["image"]).unsqueeze(0).unsqueeze(1).unsqueeze(0)
        if batch_images is None:
            batch_images = b_image
        else:
            batch_images = torch.cat([batch_images, b_image], dim=0)
    return batch_images


def get_outputs(
    model,
    batch_images,
    attention_mask,
    max_generation_length,
    min_generation_length,
    num_beams,
    length_penalty,
    input_ids,
    image_start_index_list=None,
    image_nums=None,
    bad_words_ids=None,
):
    with torch.inference_mode() and torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model.generate(
            batch_images,
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_generation_length,
            min_new_tokens=min_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            image_start_index_list=image_start_index_list,
            image_nums=image_nums,
            bad_words_ids=bad_words_ids,
        )

    outputs = outputs[:, len(input_ids[0]) :]
    return outputs


def evaluate_vqa(
    model,
    tokenizer,
    image_processor,
    batch_size,
    image_dir_path=None,
    questions_json_path=None,
    annotations_json_path=None,
    vqa_dataset="vqa",
    vis_embed_size=None,
    rank=0,
    world_size=1,
    id=0,
    is_test=True,
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor : image processor for the model
        batch_size (int): batch size
        image_dir_path (str): path to image directory
        questions_json_path (str): path to questions json file
        annotations_json_path (str): path to annotations json file
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000 samples.
        query_set_size (int, optional): size of the query set. Defaults to 2048.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        device (int, optional): device to use. Defaults to -1 (cpu).
        num_workers (int, optional): number of workers to use. Defaults to 4.
        vqa_dataset (string): type of vqa dataset: currently supports vqa, ok_vqa. Defaults to vqa.
    Returns:
        float: accuracy score
    """
    if world_size > 1:
        torch.distributed.barrier()
    if vqa_dataset == "gqa":
        eval_dataset = GQADataset()
    else:
        eval_dataset = VQADataset(
            image_dir_path=image_dir_path,
            question_path=questions_json_path,
            annotations_path=annotations_json_path,
            vqa_dataset=vqa_dataset,
            is_test=is_test,
        )
    postprocessor = OKVQAPostProcess()
    try:
        media_token_id = tokenizer("<|#image#|>", add_special_tokens=False)["input_ids"][-1]
        endofmedia_token_id = tokenizer("<|#endofimage#|>", add_special_tokens=False)["input_ids"][-1]
        pad_token_id = tokenizer(tokenizer.pad_token, add_special_tokens=False)["input_ids"][-1]
        bos_token_id = tokenizer(tokenizer.bos_token, add_special_tokens=False)["input_ids"][-1]
    except:
        pass
    def get_prompt(sample):
        return f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>Question: {sample['question'].strip()} Short answer:"
        # return f"<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>"

    model.eval().cuda()
    lang_encoder_name = model.lang_encoder.__class__.__name__.lower()
    if "peft" in lang_encoder_name:
        lang_encoder_name = model.lang_encoder.base_model.model.__class__.__name__.lower()
    predictions = []
    tokenizer.padding_side = "left"
    if world_size > 1:
        torch.distributed.barrier()
    this_tot = 0
    for ii, batch in enumerate(more_itertools.chunked(
        tqdm(eval_dataset, desc="Running inference", disable=(rank != 0)), batch_size
    )):
        if ii % world_size != rank:
            continue
        batch_images = prepare_batch_images(
            batch=batch,
            image_processor=image_processor,
        ).cuda()
        batch_text = [get_prompt(s) for s in batch]
        encodings = tokenizer(
            batch_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=2000,
        )
        input_ids = encodings["input_ids"].cuda()
        attention_mask = encodings["attention_mask"].cuda()
        skip_special_tokens = True
        image_start_index_list = ((input_ids == media_token_id).nonzero(as_tuple=True)[-1] + 1).tolist()
        image_start_index_list = [[x] for x in image_start_index_list]
        image_nums = [1] * len(input_ids)
        outputs = get_outputs(
            model=model,
            batch_images=batch_images,
            attention_mask=attention_mask,
            max_generation_length=10,
            min_generation_length=1,
            num_beams=5,
            length_penalty=-1,
            input_ids=input_ids,
            image_start_index_list=image_start_index_list,
            image_nums=image_nums,
        )
        # postprocess begin
        new_predictions = [
            out.strip().lower().strip(string.punctuation+" ") for out in tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens)
        ]
        if vqa_dataset == "ok_vqa":
            new_predictions = postprocessor._lemmatize(new_predictions)
        if model.special:
            for i in range(len(new_predictions)):
                for answer, _ in Counter(batch[i]['answers']).most_common():
                    if answer in new_predictions[i]:
                        new_predictions[i] = answer
                        break
                    if "cant" in new_predictions[i] and "no" == answer:
                        new_predictions[i] = answer
                        break
                    if "can" in new_predictions[i] and "not" not in new_predictions[i] and "cant" not in new_predictions[i] and "yes" == answer:
                        new_predictions[i] = answer
                        break
        
        this_tot += 1
        if rank == 0 and this_tot % 20 == 0:
            for i in range(1):
                tqdm.write("model output: " + new_predictions[i])

        if is_test:
            predictions.extend(
                [
                    {"answer": p, "question_id": sample["question_id"]}
                    for p, sample in zip(new_predictions, batch)
                ]
            )
        else:
            predictions.extend(
                [
                    {"answer": p, "question_id": sample["question_id"], "_question": sample["question"], "answers": sample["answers"]}
                    for p, sample in zip(new_predictions, batch)
                ]
            )
    with open(f"{vqa_dataset}_{lang_encoder_name}_results_part{rank}_{id}.json", "w") as f:
        f.write(json.dumps(predictions))
    print("save to", f"{vqa_dataset}_{lang_encoder_name}_results_part{rank}_{id}.json")

    time.sleep(10)
    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        print(f"evaluate on rank {rank}. world size is {world_size}")
        predictions = []
        for rank_i in range(world_size):
            print(f"extend rank {rank_i}")
            print("load", f"{vqa_dataset}_{lang_encoder_name}_results_part{rank_i}_{id}.json")
            predictions.extend(json.load(open(f"{vqa_dataset}_{lang_encoder_name}_results_part{rank_i}_{id}.json")))
            os.remove(f"{vqa_dataset}_{lang_encoder_name}_results_part{rank_i}_{id}.json")
        print("num:", len(predictions))
        # save the predictions to a temporary file
        random_uuid = str(uuid.uuid4())
        with open(f"{vqa_dataset}results_{random_uuid}.json", "w") as f:
            f.write(json.dumps(predictions))
        print("result saved")
        if is_test:
            exit()
        if vqa_dataset == "gqa":
            acc = compute_gqa_accuracy(predictions)
        else:
            acc = compute_vqa_accuracy(
                f"{vqa_dataset}results_{random_uuid}.json",
                questions_json_path,
                annotations_json_path,
                vqa_dataset=vqa_dataset,
            )
        print(vqa_dataset, "score:", acc, "| save to", f"{vqa_dataset}results_{random_uuid}.json")
        os.makedirs("eval_results", exist_ok=True)
        with open(os.path.join("eval_results", f"{vqa_dataset}_{model.expr_name}_{model.step_num}_{int(time.time())}_{acc}"), "w") as f:
            f.write(json.dumps(predictions, indent=2))

        # delete the temporary file
        os.remove(f"{vqa_dataset}results_{random_uuid}.json")
    else:
        time.sleep(5)
        acc = 0.0
    if world_size > 1:
        torch.distributed.barrier()
    return acc


def preprocess_conv(data):
    conversation = ""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    for idx, d in enumerate(data):
        from_str = d["from"]
        if from_str.lower() == "human":
            from_str = "Human"
        elif from_str.lower() == "gpt":
            from_str = "Assistant"
        else:
            from_str = 'unknown'
        conversation += (BEGIN_SIGNAL + from_str + ": " + d["value"] + END_SIGNAL)
    return conversation


def evaluate_refcoco(
    model,
    tokenizer,
    image_processor,
    tsvfile,
    vis_embed_size=None,
    rank=0,
    world_size=1,
    id=0,
    legacy=True,
):
    model.eval()
    media_token_id = tokenizer("<|#image#|>", add_special_tokens=False)["input_ids"][-1]
    endofmedia_token_id = tokenizer("<|#endofimage#|>", add_special_tokens=False)["input_ids"][-1]
    pad_token_id = tokenizer(tokenizer.pad_token, add_special_tokens=False)["input_ids"][-1]
    bos_token_id = tokenizer(tokenizer.bos_token, add_special_tokens=False)["input_ids"][-1]
    prebox_token_id = tokenizer("<|#prebox#|>", add_special_tokens=False)["input_ids"][-1]
    object_token_id = tokenizer("<|#object#|>", add_special_tokens=False)["input_ids"][-1]
    visual_token = "<|#visual#|>"
    previsual_token = "<|#previsual#|>"
    box_token = "<|#box#|>"
    prebox_token = "<|#prebox#|>"
    end_token = "<|#endofobject#|>"
    object_token = "<|#object#|>"
    end_of_attr_token = "<|#endofattr#|>"
    preend_of_attr_token = "<|#preendofattr#|>"
    size = image_processor.size["shortest_edge"]
    total = 0
    correct = 0
    if "refcocog" in tsvfile:
        dataset_name = "refcocog"
    elif "refcocoplus" in tsvfile:
        dataset_name = "refcocoplus"
    else:
        dataset_name = "refcoco"
    with open(tsvfile, "r") as f:
        lines = f.readlines()
        pbar = tqdm(lines)
        for ii, line in enumerate(pbar):
            if ii % world_size != rank:
                continue
            total += 1
            line = line.rstrip()
            uniq_id, image_id, text, region_coord, image = line.split("\t")

            image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")
            gt_box = np.array(list(map(float, region_coord.split(","))))
            width = image.width
            height = image.height
            image = image.resize((size, size))
            gt_box = gt_box / np.array([width, height, width, height]) * size
            batch_images = preprocess_image(image, image_processor).unsqueeze(0).unsqueeze(1).unsqueeze(0)
            text = text.rstrip('.').strip().replace('"', '').lower()
            conversation = [
                {
                    "from": "human",
                    "value": f"Please provide the bounding box coordinate of the region this sentence describes: {text}.",
                },
                {
                    "from": "gpt",
                    "value": object_token + text + end_token + visual_token
                }
            ]
            text = preprocess_conv(conversation).strip()
            prompt = [f"<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>{text}"]
            encodings = tokenizer(
                prompt,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=2000,
            )
            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]
            # attention_mask[input_ids == prebox_token_id] = 0
            image_start_index_list = ((input_ids == media_token_id).nonzero(as_tuple=True)[-1] + 1).tolist()
            image_start_index_list = [[x] for x in image_start_index_list]
            image_nums = [1] * len(input_ids)
            vision_x = batch_images.to("cuda")
            lang_x = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
            model.debug_id = 0
            with torch.no_grad() and torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask,
                    labels=None,
                    image_nums=image_nums,
                    image_start_index_list=image_start_index_list,
                    added_bbox_list=None,
                    add_box=False,
                )
            boxes = outputs["boxes"]
            scores = outputs["scores"]
            if not legacy:
                boxes = boxes[scores >= scores[0]*0.5]
                scores = scores[scores >= scores[0]*0.5]
                del outputs
                text = text.lower().strip()
                if text.split(" ")[0] not in ["a", "an", "the", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "several", "some"]:
                    text = "a " + text
                losses = []
                for box, score in zip(boxes, scores):
                    this_prompt = [f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>There is<|#object#|><|#previsual#|><|#prebox#|><|#object#|> {text}"]
                    encodings = tokenizer(
                        this_prompt,
                        padding="longest",
                        truncation=True,
                        return_tensors="pt",
                        max_length=2000,
                    )
                    input_ids = encodings["input_ids"]
                    attention_mask = encodings["attention_mask"]
                    image_start_index_list = ((input_ids == media_token_id).nonzero(as_tuple=True)[-1] + 1).tolist()
                    image_start_index_list = [[x] for x in image_start_index_list]
                    image_nums = [1] * len(input_ids)
                    vision_x = batch_images.cuda()
                    lang_x = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    added_bbox_list = [torch.tensor(box / size).cuda().unsqueeze(0).clamp(0, 0.99)]
                    labels = lang_x.clone()
                    start_idx = (lang_x == object_token_id).nonzero()[-1, -1]
                    labels[0, :start_idx+1] = -100
                    with torch.no_grad() and torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = model(
                            vision_x=vision_x,
                            lang_x=lang_x,
                            attention_mask=attention_mask,
                            labels=labels,
                            image_nums=image_nums,
                            image_start_index_list=image_start_index_list,
                            added_bbox_list=added_bbox_list,
                            add_box=True,
                        )
                        # print(tokenizer.decode(outputs.logits[0, start_idx].sort(descending=True).indices[:10]))
                        loss = outputs.loss.detach().cpu()
                        losses.append((loss.sum() / (loss != 0).sum()).item())
                        del outputs
                chosen_idx = np.array(losses).argmin()
                pred_box = boxes[chosen_idx]
                # if chosen_idx != 0:
                #     tqdm.write(f"{text}|{chosen_idx}|{scores[chosen_idx]}")
            else:
                pred_box = boxes[0] if boxes is not None and len(boxes) > 0 else [0, 0, size, size]
            iou = get_iou(pred_box, gt_box)
            if iou >= 0.5:
                correct += 1
            pbar.set_description(f"iou: {iou:.2f} score: {correct / total:.4f}")

    with open(f"{dataset_name}_results_part{rank}_{id}.json", "w") as f:
        f.write(json.dumps([total, correct]))
    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        total = 0
        correct = 0
        print(f"evaluate on rank {rank}. world size is {world_size}")
        for rank_i in range(world_size):
            [total_part, correct_part] = json.load(open(f"{dataset_name}_results_part{rank_i}_{id}.json"))
            os.remove(f"{dataset_name}_results_part{rank_i}_{id}.json")
            total += total_part
            correct += correct_part
        score = correct / total
        print("score:", score)
        with open(os.path.join("eval_results", f"{dataset_name}_{model.expr_name}_{model.step_num}_{int(time.time())}_{score}"), "w") as f:
            pass
    else:
        score = 0.0
    if world_size > 1:
        torch.distributed.barrier()
    return score


def preprocess_visual_info(Text):
    text = Text.split(" ")
    for is_idx, t in enumerate(text):
        if t == "is":
            break
    the_idx = is_idx
    while text[the_idx] != "the":
        the_idx -= 1
    obj_A = " ".join(text[the_idx+1:is_idx])
    second_the_idx = len(text) - 1
    while text[second_the_idx] != "the":
        second_the_idx -= 1
    obj_B = " ".join(text[second_the_idx+1:])
    relation = " ".join(text[is_idx+1:second_the_idx])
    visual_obj_A = f"<|#object#|>the {obj_A}<|#endofobject#|><|#visual#|><|#box#|><|#endofobject#|>"
    visual_obj_B = f"<|#object#|><|#previsual#|><|#prebox#|><|#object#|>the {obj_B}<|#endofobject#|>"
    Text = f"{visual_obj_A} is {relation} {visual_obj_B}"
    return Text, obj_A, visual_obj_A, obj_B, visual_obj_B, relation


def get_bbox(visual_box_list, batch_images, prompt, model, tokenizer, media_token_id, prebox_token_id, debug=False, return_all=False):
    assert isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str)
    encodings = tokenizer(
        prompt,
        padding="longest",
        truncation=True,
        return_tensors="pt",
        max_length=2000,
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    image_start_index_list = ((input_ids == media_token_id).nonzero(as_tuple=True)[-1] + 1).tolist()
    image_start_index_list = [[x] for x in image_start_index_list]
    image_nums = [1] * len(input_ids)
    vision_x = batch_images.cuda()
    lang_x = input_ids.cuda()
    attention_mask = attention_mask.cuda()

    model.debug_id = 0
    with torch.inference_mode() and torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
            labels=None,
            image_nums=image_nums,
            image_start_index_list=image_start_index_list,
            added_bbox_list=visual_box_list,
            add_box=visual_box_list is not None,
            relations=None,
            debug_mode=False,
        )
    boxes = outputs["boxes"]
    scores = outputs["scores"]
    if debug:
        import pdb; pdb.set_trace()
    if return_all:
        return boxes, scores
    if len(scores) == 0:
        return None, None
    else:
        return boxes[scores.argmax()], scores.max()


def evaluate_aro(
    model,
    tokenizer,
    image_processor,
    vis_embed_size=None,
    rank=0,
    world_size=1,
    id=0,
    add_visual=True,
    subset=False,
    choose_left_right=False,
):
    # os.makedirs(f"visualization/aro_results_{id}", exist_ok=True)
    dataset_name = "aro"
    media_token_id = tokenizer("<|#image#|>", add_special_tokens=False)["input_ids"][-1]
    box_token_id = tokenizer("<|#box#|>", add_special_tokens=False)["input_ids"][-1]
    endofobject_token_id = tokenizer("<|#endofobject#|>", add_special_tokens=False)["input_ids"][-1]
    endofattr_token_id = tokenizer("<|#endofattr#|>", add_special_tokens=False)["input_ids"][-1]
    endofmedia_token_id = tokenizer("<|#endofimage#|>", add_special_tokens=False)["input_ids"][-1]
    visual_token_id = tokenizer("<|#visual#|>", add_special_tokens=False)["input_ids"][-1]
    previsual_token_id = tokenizer("<|#previsual#|>", add_special_tokens=False)["input_ids"][-1]
    prebox_token_id = tokenizer("<|#prebox#|>", add_special_tokens=False)["input_ids"][-1]
    model.eval().cuda()
    total = 0
    n_top1 = 0
    n_top5 = 0
    from open_flamingo.eval.dataset_zoo import VG_Relation, VG_Attribution
    vgr_dataset = VG_Relation(image_preprocess=None, download=True, root_dir="/gpfs/u/home/LMCG/LMCGljnn/scratch/code/vision-language-models-are-bows/data")
    if subset:
        subset_idx = json.load(open("aro_subset.json"))
        pbar = tqdm(subset_idx, disable=(rank != 0))
    else:
        pbar = tqdm(vgr_dataset, disable=(rank != 0))
    for ii, sample in enumerate(pbar):
        if subset:
            ORI_IDX = int(sample)
            sample = vgr_dataset[sample]
        if ii % world_size != rank:
            continue
        image = sample["image_options"][0]
        # image = Image.open("/gpfs/u/home/LMCG/LMCGljnn/scratch/code/multimodal2/yolo.png").convert("RGB")
        image = image.resize((224, 224))

        text = sample["caption_options"][1] # 1 is true caption
        # text = "the dog is sitting on the floor" if idx == 1 else "the floor is sitting on the dog"
        batch_images = image_processor(image).unsqueeze(0).unsqueeze(1).unsqueeze(0)
        text, obj_A, visual_obj_A, obj_B, visual_obj_B, relation = preprocess_visual_info(text)


        first_text = f"<|#object#|>the {obj_A}<|#endofobject#|><|#visual#|>"
        prompt = [f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>{first_text}"]
        first_box, first_score = get_bbox(None, batch_images, prompt, model, tokenizer, media_token_id, prebox_token_id, return_all=False)

        if first_box is None:
            text_A = "the " + obj_A
            added_bbox_list = None
        else:
            text_A = visual_obj_A
            added_bbox_list = [torch.tensor(first_box).unsqueeze(0).cuda() / 224]

        prompt = [f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>{text_A} is {relation}<|#object#|><|#previsual#|>"]
        pre_boxes, pre_scores = get_bbox(added_bbox_list, batch_images, prompt, model, tokenizer, media_token_id, 
        prebox_token_id, return_all=True)

        if pre_boxes is None:
            pre_boxes = [np.array([0.0, 0.0, 223.0, 223.0])]
            pre_scores = [1.0]

        logits_list = []
        # pre_boxes = [pre_boxes[0]]
        # pre_scores = [pre_scores[0]]
        for pre_box, pre_score in zip(pre_boxes, pre_scores):
            prompt = [f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>{text_A} is {relation}<|#object#|><|#previsual#|><|#prebox#|><|#object#|> the {obj_B}<|#endofobject#|>"]

            encodings = tokenizer(
                prompt,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]
            image_start_index_list = ((input_ids == media_token_id).nonzero(as_tuple=True)[-1] + 1).tolist()
            image_start_index_list = [[x] for x in image_start_index_list]
            image_nums = [1] * len(input_ids)
            vision_x = batch_images.cuda()
            lang_x = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = lang_x.clone()
            added_bbox_list = None
            if add_visual:
                added_bbox_list = []
                if first_box is not None:
                    added_bbox_list.append(torch.tensor(first_box).unsqueeze(0).cuda().float() / 224)
                if pre_box is not None:
                    added_bbox_list.append(torch.tensor(pre_box).unsqueeze(0).cuda().float() / 224)
            if added_bbox_list is not None and len(added_bbox_list) == 0:
                added_bbox_list = None

            with torch.cuda.amp.autocast(dtype=torch.float16) and torch.no_grad():
                outputs = model(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask,
                    labels=labels,
                    image_nums=image_nums,
                    image_start_index_list=image_start_index_list,
                    added_bbox_list=added_bbox_list,
                    add_box=added_bbox_list is not None,
                    relations=None,
                )
            logits_list.append([pre_score, outputs.logits])
        pre_scores = np.array([x[0] for x in logits_list])
        final_probs = 0.0
        for score, (_, logits) in zip(pre_scores, logits_list):
            final_probs += score * logits.softmax(-1)
        assert input_ids.shape[:2] == final_probs.shape[:2]
        _rank, is_top1, is_top5 = is_correct(input_ids, final_probs, tokenizer, obj_B, topk=5)
        if is_top1:
            n_top1 += 1
        if is_top5:
            n_top5 += 1
        total += 1
        pbar.set_description(f"acc@top1: {n_top1 / total:.4f} | acc@top5: {n_top5 / total:.4f} | {_rank}")


    with open(f"{dataset_name}_results_part{rank}_{id}.json", "w") as f:
        f.write(json.dumps([total, n_top1, n_top5]))
    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        total = 0
        n_top1 = 0
        n_top5 = 0
        print(f"evaluate on rank {rank}. world size is {world_size}")
        for rank_i in range(world_size):
            [total_part, n_top1_part, n_top5_part] = json.load(open(f"{dataset_name}_results_part{rank_i}_{id}.json"))
            os.remove(f"{dataset_name}_results_part{rank_i}_{id}.json")
            total += total_part
            n_top1 += n_top1_part
            n_top5 += n_top5_part
        acc_top1 = n_top1 / total
        acc_top5 = n_top5 / total
        print("acc_top1:", acc_top1, "acc_top5:", acc_top5, "total:", total)
        with open(os.path.join("eval_results", f"{dataset_name}_{model.expr_name}_{model.step_num}_{int(time.time())}_{acc_top1}_{acc_top5}_{total}_{subset}"), "w") as f:
            pass
    else:
        score = 0.0
    if world_size > 1:
        torch.distributed.barrier()
    return score


def evaluate_exp(
    model,
    tokenizer,
    image_processor,
    vis_embed_size=None,
    rank=0,
    world_size=1,
    id=0,
    add_visual=True,
):
    media_token_id = tokenizer("<|#image#|>", add_special_tokens=False)["input_ids"][-1]
    box_token_id = tokenizer("<|#box#|>", add_special_tokens=False)["input_ids"][-1]
    endofobject_token_id = tokenizer("<|#endofobject#|>", add_special_tokens=False)["input_ids"][-1]
    endofattr_token_id = tokenizer("<|#endofattr#|>", add_special_tokens=False)["input_ids"][-1]
    endofmedia_token_id = tokenizer("<|#endofimage#|>", add_special_tokens=False)["input_ids"][-1]
    visual_token_id = tokenizer("<|#visual#|>", add_special_tokens=False)["input_ids"][-1]
    previsual_token_id = tokenizer("<|#previsual#|>", add_special_tokens=False)["input_ids"][-1]
    prebox_token_id = tokenizer("<|#prebox#|>", add_special_tokens=False)["input_ids"][-1]
    size = image_processor.size["shortest_edge"]
    model.eval()
    # "/gpfs/u/home/LMCG/LMCGljnn/scratch-shared/cdl/tmp_img/chat_vis/chat19.png"
    image_path = input("Please enter the image path: ")
    image = Image.open(image_path).convert("RGB")
    image = image.resize((size, size))
    print(f"image size: {image.size}")
    batch_images = preprocess_image(image, image_processor).unsqueeze(0).unsqueeze(1).unsqueeze(0).to("cuda")
    conversation = []
    human_sentence = None
    while True:
        human_sentence = input("### Human: ")
        if human_sentence == "#end#":
            break
        conversation.append({
            "from": "human",
            "value": human_sentence,
        })
        conversation.append({
            "from": "gpt",
            "value": "",
        })
        text = preprocess_conv(conversation).strip()
        caption = f"<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>{text}"
        encodings = tokenizer(
            caption,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"].to("cuda")
        attention_mask = encodings["attention_mask"].to("cuda")
        image_start_index_list = ((input_ids == media_token_id).nonzero(as_tuple=True)[-1] + 1).tolist()
        image_start_index_list = [[x] for x in image_start_index_list]
        image_nums = [1] * len(input_ids)
        with torch.no_grad() and torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model.generate(
                batch_images,
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                # min_new_tokens=8,
                num_beams=1,
                image_start_index_list=image_start_index_list,
                image_nums=image_nums,
            )
        print(f"### Assistant: {tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True).strip()}")


if __name__ == "__main__":
    main()

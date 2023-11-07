import json
import webdataset as wds
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
import os
import time
import cv2
import random
import math
from open_flamingo.eval.task.utils import (
    get_object_from_text,
    is_correct,
    _eval_text_image,
    get_bbox,
    get_iou,
)
DATASET = "/gpfs/u/home/LMCG/LMCGljnn/scratch/code/COLA/data/COLA_multiobjects_matching_benchmark.json"
VG_ROOT = "/gpfs/u/home/LMCG/LMCGljnn/scratch/datasets/raw/vg/VG_100K"


def get_score(image, text, model, tokenizer, image_processor, vis_embed_size):
    media_token_id = tokenizer("<|#image#|>", add_special_tokens=False)["input_ids"][-1]
    prebox_token_id = tokenizer("<|#prebox#|>", add_special_tokens=False)["input_ids"][-1]
    object_token_id = tokenizer("<|#object#|>", add_special_tokens=False)["input_ids"][-1]
    text = text.split("#")
    obj_A = text[0].strip().split(" ")
    relation = text[1].strip()
    obj_B = text[2].strip().split(" ")
    if "computer mouse" not in text[0].strip():
        attrAs = obj_A[:-1]
        nounA = obj_A[-1]
    else:
        attrAs = obj_A[:-2]
        nounA = " ".join(obj_A[-2:])
    if "computer mouse" not in text[2].strip():
        attrBs = obj_B[:-1]
        nounB = obj_B[-1]
    else:
        attrBs = obj_B[:-2]
        nounB = " ".join(obj_B[-2:])
    batch_images = image_processor(image).unsqueeze(0).unsqueeze(1).unsqueeze(0)
    prompt1 = [f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|><|#object#|>the {nounA}<|#endofobject#|><|#visual#|>"]
    boxes, scores = get_bbox(None, batch_images, prompt1, model, tokenizer, media_token_id, prebox_token_id, return_all=True)
    box_ppl = []
    box_attr_losses = []
    for box in boxes:
        losses = []
        for attrA in attrAs:
            prompt2 = [f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|><|#object#|><|#previsual#|><|#prebox#|><|#object#|> the {attrA} {nounA}"]
            encodings = tokenizer(
                prompt2,
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
            start_idx = (labels == object_token_id).nonzero()[-1, -1]
            labels[0, :start_idx+1] = -100
            added_bbox_list = [torch.tensor(box / 224.0).cuda().unsqueeze(0)]
            with torch.cuda.amp.autocast(dtype=torch.float16) and torch.no_grad():
                outputs = model(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask,
                    labels=labels,
                    image_nums=image_nums,
                    image_start_index_list=image_start_index_list,
                    added_bbox_list=added_bbox_list,
                )
            loss = outputs.loss
            loss = (loss.sum() / (loss != 0).sum()).item()
            losses.append(loss)
        avg_ppl = np.array(losses).mean()
        box_ppl.append(avg_ppl)
        box_attr_losses.append(losses)
    fit_idx = np.array(box_ppl).argmin()
    fit_box = boxes[fit_idx]
    fit_attr = attrAs[np.array(box_attr_losses[fit_idx]).argmin()]
    first_ppl = min(box_ppl)
    prompt3 = [f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|><|#object#|>the {fit_attr} {nounA}<|#endofobject#|><|#visual#|><|#box#|><|#endofobject#|> is {relation}<|#object#|><|#previsual#|>"]
    boxes, scores = get_bbox([torch.tensor(fit_box / 224).cuda().unsqueeze(0)], batch_images, prompt3, model, tokenizer, media_token_id, prebox_token_id, return_all=True)
    box_ppl = []
    for box in boxes:
        losses = []
        for attrB in attrBs:
            prompt4 = [f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|><|#object#|>the {fit_attr} {nounA}<|#endofobject#|><|#visual#|><|#box#|><|#endofobject#|> is {relation}<|#object#|><|#previsual#|><|#prebox#|><|#object#|> the {attrB} {nounB}"]
            encodings = tokenizer(
                prompt4,
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
            start_idx = (labels == object_token_id).nonzero()[-1, -1]
            labels[0, :start_idx+1] = -100
            added_bbox_list = [torch.tensor(fit_box / 224.0).cuda().unsqueeze(0), torch.tensor(box / 224.0).cuda().unsqueeze(0)]
            with torch.cuda.amp.autocast(dtype=torch.float16) and torch.no_grad():
                outputs = model(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask,
                    labels=labels,
                    image_nums=image_nums,
                    image_start_index_list=image_start_index_list,
                    added_bbox_list=added_bbox_list,
                )
            loss = outputs.loss
            loss = (loss.sum() / (loss != 0).sum()).item()
            losses.append(loss)
        avg_ppl = np.array(losses).mean()
        box_ppl.append(avg_ppl)
    second_ppl = (np.array(box_ppl) * np.array(scores)).sum() / sum(scores)
    return (first_ppl + second_ppl) / 2


def evaluate_cola(
    model,
    tokenizer,
    image_processor,
    vis_embed_size=None,
    rank=0,
    world_size=1,
    id=0,
    debug=False,
):
    dataset_name = "cola"
    dataset = json.load(open(DATASET))
    model = model.cuda().eval()
    correct = 0
    total = 0
    pbar = tqdm(dataset, disable=(rank != 0))
    for ii, sample in enumerate(pbar):
        if ii % world_size != rank:
            continue
        image1 = Image.open(os.path.join(VG_ROOT, os.path.basename(sample[0]))).convert("RGB").resize((224, 224))
        text1 = sample[1]
        image2 = Image.open(os.path.join(VG_ROOT, os.path.basename(sample[2]))).convert("RGB").resize((224, 224))
        text2 = sample[3]
        score11 = -get_score(image1, text1, model, tokenizer, image_processor, vis_embed_size)
        score12 = -get_score(image1, text2, model, tokenizer, image_processor, vis_embed_size)
        score21 = -get_score(image2, text1, model, tokenizer, image_processor, vis_embed_size)
        score22 = -get_score(image2, text2, model, tokenizer, image_processor, vis_embed_size)
        if score11 > score21 and score22 > score12:
            correct += 1
        total += 1
        pbar.set_description(f"{correct / total:.2f}")

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
        with open(os.path.join("eval_results", f"{dataset_name}_{model.expr_name}_{model.step_num}_{int(time.time())}_{score}_{total}"), "w") as f:
            pass
    else:
        score = 0.0
    if world_size > 1:
        torch.distributed.barrier()
    return score


if __name__ == "__main__":
    evaluate_cola(None, None, None)

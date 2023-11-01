import json
import time
import torch
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from open_flamingo.eval.task.utils import get_iou
import os


def evaluate_refcoco(
    model,
    tokenizer,
    image_processor,
    tsvfile,
    vis_embed_size=None,
    rank=0,
    world_size=1,
    id=0,
    legacy=False,
):
    model.eval()
    media_token_id = tokenizer("<|#image#|>", add_special_tokens=False)["input_ids"][-1]
    object_token_id = tokenizer("<|#object#|>", add_special_tokens=False)["input_ids"][-1]
    try:
        size = image_processor.size["shortest_edge"]
    except AttributeError:
        size = image_processor.transforms[0].size
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
        pbar = tqdm(lines, disable=(rank != 0))
        for ii, line in enumerate(pbar):
            if ii % world_size != rank:
                continue
            line = line.rstrip()
            uniq_id, image_id, text, region_coord, image = line.split("\t")

            image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")
            gt_box = np.array(list(map(float, region_coord.split(","))))
            width = image.width
            height = image.height
            image = image.resize((size, size))
            gt_box = gt_box / np.array([width, height, width, height]) * 224
            batch_images = image_processor(image).unsqueeze(0).unsqueeze(1).unsqueeze(0)
            text = text.rstrip('.').strip().replace('"', '').capitalize()
            prompt = [f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|><|#object#|>{text}<|#endofobject#|><|#visual#|>"]

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
            vision_x = batch_images.to("cuda")
            lang_x = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")

            with torch.no_grad() and torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask,
                    labels=None,
                    image_nums=image_nums,
                    image_start_index_list=image_start_index_list,
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
                    vision_x = batch_images.to("cuda")
                    lang_x = input_ids.to("cuda")
                    attention_mask = attention_mask.to("cuda")
                    added_bbox_list = [torch.tensor(box / 224).to("cuda").unsqueeze(0).clamp(0, 0.99)]
                    labels = lang_x.clone()
                    start_idx = (lang_x == object_token_id).nonzero()[-1, -1]
                    labels[0, :start_idx + 1] = -100
                    with torch.no_grad() and torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = model(
                            vision_x=vision_x,
                            lang_x=lang_x,
                            attention_mask=attention_mask,
                            labels=labels,
                            image_nums=image_nums,
                            image_start_index_list=image_start_index_list,
                            added_bbox_list=added_bbox_list,
                        )
                        loss = outputs.loss.detach().cpu()
                        losses.append((loss.sum() / (loss != 0).sum()).item())
                        del outputs
                chosen_idx = np.array(losses).argmin()
                pred_box = boxes[chosen_idx]
            else:
                pred_box = boxes[0] if boxes is not None and len(boxes) > 0 else [0, 0, size, size]
            iou = get_iou(pred_box, gt_box)
            if iou >= 0.5:
                correct += 1
            total += 1
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

import torch
from tqdm import tqdm
from open_flamingo.eval.dataset_zoo import VG_Relation
from open_flamingo.eval.task.utils import preprocess_visual_info, get_bbox, is_correct
import json
import numpy as np
import os
import time


def evaluate_aro(
    model,
    tokenizer,
    image_processor,
    vis_embed_size=None,
    rank=0,
    world_size=1,
    id=0,
    dataset_root: str = "~/scratch/code/vision-language-models-are-bows/data"
):
    # os.makedirs(f"visualization/aro_results_{id}", exist_ok=True)
    dataset_name = "aro"
    media_token_id = tokenizer("<|#image#|>", add_special_tokens=False)["input_ids"][-1]
    prebox_token_id = tokenizer("<|#prebox#|>", add_special_tokens=False)["input_ids"][-1]
    model.eval().cuda()
    total = 0
    n_top1 = 0
    n_top5 = 0
    vgr_dataset = VG_Relation(image_preprocess=None, download=True, root_dir=dataset_root)
    pbar = tqdm(vgr_dataset, disable=(rank != 0))
    for ii, sample in enumerate(pbar):
        if ii % world_size != rank:
            continue
        image = sample["image_options"][0]
        image = image.resize((224, 224))

        text = sample["caption_options"][1]
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
        pre_boxes, pre_scores = get_bbox(added_bbox_list, batch_images, prompt, model, tokenizer, media_token_id, prebox_token_id, return_all=True)

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
        with open(os.path.join("eval_results", f"{dataset_name}_{model.expr_name}_{model.step_num}_{int(time.time())}_{acc_top1}_{acc_top5}_{total}"), "w") as f:
            pass
    else:
        score = 0.0
    if world_size > 1:
        torch.distributed.barrier()
    return score

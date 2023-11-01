from torch.utils.data import Dataset
import json
from PIL import Image
import os
import torch
import more_itertools
from tqdm import tqdm
import time
from vqa_metric import compute_gqa_accuracy
import string
import uuid
import numpy as np
import cv2
from open_flamingo.eval.task.utils import get_bbox

class GQADataset(Dataset):
    def __init__(
        self,
        image_dir_path="/gpfs/u/home/LMCG/LMCGljnn/scratch/datasets/raw/gqa/images",
        annotations_path="/gpfs/u/home/LMCG/LMCGljnn/scratch/datasets/raw/gqa/testdev_balanced_questions.json",
    ):
        annotations = json.load(open(annotations_path))
        self.questions = []
        self.answers = []
        self.image_paths = []
        self.question_ids = []
        for anno_id in annotations:
            question = annotations[anno_id]["question"]
            imageId = annotations[anno_id]["imageId"]
            answer = annotations[anno_id]["answer"]
            self.questions.append(question)
            self.answers.append(answer)
            self.image_paths.append(os.path.join(image_dir_path, "{}.jpg".format(imageId)))
            self.question_ids.append(anno_id)
            # print(annotations[anno_id]["types"])
        self.vqa_dataset = "gqa"

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        question_id = self.question_ids[idx]
        answer = self.answers[idx]
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        return {
            "image": image,
            "question": question,
            "answers": answer,
            "question_id": question_id,
        }


def prepare_batch_images(batch, image_processor):
    batch_images = None
    for b in batch:
        b_image = image_processor(b["image"]).unsqueeze(0).unsqueeze(1).unsqueeze(0)
        if batch_images is None:
            batch_images = b_image
        else:
            batch_images = torch.cat([batch_images, b_image], dim=0)
    return batch_images


def evaluate_gqa(
    model,
    tokenizer,
    image_processor,
    batch_size=1,
    vis_embed_size=None,
    rank=0,
    world_size=1,
    id=0,
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
    assert batch_size == 1
    vqa_dataset = "gqa"
    eval_dataset = GQADataset()
    object_token_id = tokenizer("<|#object#|>", add_special_tokens=False)["input_ids"][-1]
    endofobject_token_id = tokenizer("<|#endofobject#|>", add_special_tokens=False)["input_ids"][-1]
    prebox_token_id = tokenizer("<|#prebox#|>", add_special_tokens=False)["input_ids"][-1]
    media_token_id = tokenizer("<|#image#|>", add_special_tokens=False)["input_ids"][-1]
    endofmedia_token_id = tokenizer("<|#endofimage#|>", add_special_tokens=False)["input_ids"][-1]
    pad_token_id = tokenizer(tokenizer.pad_token, add_special_tokens=False)["input_ids"][-1]
    bos_token_id = tokenizer(tokenizer.bos_token, add_special_tokens=False)["input_ids"][-1]
    def get_prompt(sample):
        return f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>Question: {sample['question'].strip()} Short answer:"
    model.eval().cuda()
    lang_encoder_name = model.lang_encoder.__class__.__name__.lower()
    predictions = []
    if batch_size != 1:
        tokenizer.padding_side = "left"
    if world_size > 1:
        torch.distributed.barrier()
    this_tot = 0
    for ii, batch in enumerate(more_itertools.chunked(
        tqdm(eval_dataset, desc="Running inference", disable=(rank != 0)), batch_size,
    )):
        if ii % world_size != rank:
            continue
        batch[0]["image"] = batch[0]["image"].resize((224, 224))
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
        image_start_index_list = ((input_ids == media_token_id).nonzero(as_tuple=True)[-1] + 1).tolist()
        image_start_index_list = [[x] for x in image_start_index_list]
        image_nums = [1] * len(input_ids)
        with torch.inference_mode() and torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model.generate(
                batch_images,
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                min_length=1,
                num_beams=1,
                # length_penalty=0,
                image_start_index_list=image_start_index_list,
                image_nums=image_nums,
                added_bbox_list=None,
                return_dict_in_generate=True,
                output_scores=True,
            )
        scores = outputs.scores
        outputs = outputs.sequences[:, len(input_ids[0]) :]
        if object_token_id in scores[0][0].sort(descending=True).indices[:5]:
            sample = batch[0]
            # print("="*80)
            # print("sample:", batch, scores[0][0].sort(descending=True).indices[:10].tolist().index(object_token_id))
            prompt1 = [f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>Question: {sample['question'].strip()} Short answer:<|#object#|><|#previsual#|>"]
            boxes, scores = get_bbox(None, batch_images, prompt1, model, tokenizer, media_token_id, prebox_token_id, return_all=True)
            # open_cv_image = np.array(sample["image"])
            # open_cv_image = open_cv_image[:, :, ::-1].copy()
            # cv2.imwrite(f"Atest_ori.png", open_cv_image)
            # open_cv_image = cv2.rectangle(open_cv_image, boxes[0][:2].astype(int), boxes[0][2:].astype(int), (0, 255, 0), 2)
            # print(scores)
            # cv2.imwrite(f"Atest.png", open_cv_image)
            if boxes is not None and len(boxes) > 0:
                prompt2 = [f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>Question: {sample['question'].strip()} Short answer: it is<|#object#|><|#previsual#|><|#prebox#|><|#object#|> a"]
                encodings = tokenizer(
                    prompt2,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=2000,
                )
                input_ids = encodings["input_ids"].cuda()
                attention_mask = encodings["attention_mask"].cuda()
                image_start_index_list = ((input_ids == media_token_id).nonzero(as_tuple=True)[-1] + 1).tolist()
                image_start_index_list = [[x] for x in image_start_index_list]
                image_nums = [1] * len(input_ids)
                added_bbox_list = [torch.tensor(boxes[0]/224.0).cuda().unsqueeze(0).clamp(0, 0.99)]
                with torch.inference_mode() and torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = model.generate(
                        batch_images,
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=10,
                        min_length=1,
                        num_beams=1,
                        image_start_index_list=image_start_index_list,
                        image_nums=image_nums,
                        added_bbox_list=added_bbox_list,
                        eos_token_id=(endofobject_token_id),
                    )
                outputs = outputs[:, len(input_ids[0]) :]
                # print("previsual===>{}".format(tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower().strip(string.punctuation+" ")))

        # postprocess begin
        new_predictions = [
            out.strip().lower().strip(string.punctuation+" ") for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]
        this_tot += 1
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
            print("load", f"{vqa_dataset}_{lang_encoder_name}_results_part{rank_i}_{id}.json")
            predictions.extend(json.load(open(f"{vqa_dataset}_{lang_encoder_name}_results_part{rank_i}_{id}.json")))
            os.remove(f"{vqa_dataset}_{lang_encoder_name}_results_part{rank_i}_{id}.json")
        print("num:", len(predictions))
        # save the predictions to a temporary file
        random_uuid = str(uuid.uuid4())
        with open(f"{vqa_dataset}results_{random_uuid}.json", "w") as f:
            f.write(json.dumps(predictions, indent=4))

        acc = compute_gqa_accuracy(predictions)
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

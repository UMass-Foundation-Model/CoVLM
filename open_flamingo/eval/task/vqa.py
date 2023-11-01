import torch
from open_flamingo.eval.eval_datasets import VQADataset
from open_flamingo.eval.task.utils import prepare_batch_images, get_outputs
from tqdm import tqdm
import more_itertools
import logging
import string
import time
import json
import os
import uuid
from open_flamingo.eval.vqa_metric import compute_vqa_accuracy


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
    if world_size > 1:
        torch.distributed.barrier()
    eval_dataset = VQADataset(
        image_dir_path=image_dir_path,
        question_path=questions_json_path,
        annotations_path=annotations_json_path,
        vqa_dataset=vqa_dataset,
        is_test=is_test,
    )
    postprocessor = OKVQAPostProcess()
    media_token_id = tokenizer("<|#image#|>", add_special_tokens=False)["input_ids"][-1]

    def get_prompt(sample):
        return f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>Question: {sample['question'].strip()} Short answer:"

    model.eval().cuda()
    lang_encoder_name = model.lang_encoder.__class__.__name__.lower()
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
            out.strip().lower().strip(string.punctuation + " ") for out in tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens)
        ]
        if vqa_dataset == "ok_vqa":
            new_predictions = postprocessor._lemmatize(new_predictions)
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

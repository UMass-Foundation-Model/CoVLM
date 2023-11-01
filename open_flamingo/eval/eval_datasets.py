import json
import os

from PIL import Image
from torch.utils.data import Dataset


class COCOFlickrDataset(Dataset):
    def __init__(
        self,
        image_dir_path,
        annotations_path,
        is_flickr=False,
    ):
        self.image_dir_path = image_dir_path
        self.annotations = json.load(open(annotations_path))["annotations"]
        self.is_flickr = is_flickr

    def __len__(self):
        return len(self.annotations)

    def get_img_path(self, idx):
        if self.is_flickr:
            return f"{self.image_dir_path}/{self.annotations[idx]['image_id']}.jpg"
        else:
            return f"{self.image_dir_path}/{self.annotations[idx]['image_id']:012d}.jpg"

    def __getitem__(self, idx):
        image = Image.open(self.get_img_path(idx))
        caption = self.annotations[idx]["caption"]
        return {
            "image": image,
            "caption": caption,
            "image_id": self.annotations[idx]["image_id"],
        }


class VQADataset(Dataset):
    def __init__(
        self,
        image_dir_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/train2014/",
        question_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json",
        annotations_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/v2_mscoco_train2014_annotations.json",
        vqa_dataset="vqa",
        is_test=False,
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        self.answers = json.load(open(annotations_path, "r"))["annotations"] if annotations_path != "none" else None
        self.image_dir_path = image_dir_path
        self.vqa_dataset = vqa_dataset
        self.is_test = is_test

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.vqa_dataset == "vqa":
            if self.is_test:
                sufix = "test2015"
            else:
                sufix = "val2014"
            return os.path.join(
                self.image_dir_path, f"COCO_{sufix}_{question['image_id']:012d}.jpg"
            )
        elif self.vqa_dataset == "ok_vqa":
            return os.path.join(
                self.image_dir_path, f"COCO_val2014_{question['image_id']:012d}.jpg"
            )
        else:
            raise Exception(f"Unknown VQA dataset {self.vqa_dataset}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx] if self.answers is not None else None
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        return {
            "image": image,
            "question": question["question"],
            "answers": [a["answer"] for a in answers["answers"]] if answers is not None else None,
            "question_id": question["question_id"],
        }


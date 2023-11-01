import torch
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from PIL import Image

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, cpu_only=False):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(
                pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


class caption_grounder():
    def __init__(self, config_file, checkpoint_path, box_threshold=0.35, text_threshold=0.25, cpu_only=False):
        self.cpu_only = cpu_only
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.ground_model = self.load_model(
		    config_file, checkpoint_path, cpu_only=cpu_only)

    def load_model(self, model_config_path, model_checkpoint_path, cpu_only=False):
        args = SLConfig.fromfile(model_config_path)
        args.device = "cuda" if not cpu_only else "cpu"
        model = build_model(args)
        if model_checkpoint_path is not None:
            checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
            load_res = model.load_state_dict(
                clean_state_dict(checkpoint["model"]), strict=False)
            print(load_res)
        _ = model.eval()
        return model

    def load_image(self, image_path=None, image_pil=None):
        if image_pil is None:
            image_pil = Image.open(image_path).convert("RGB")
        transform = T.Compose([
				T.ToTensor(),
				T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True, cpu_only=False, return_raw=False):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = "cuda" if not cpu_only else "cpu"
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        if return_raw:
            return logits_filt, boxes_filt
        
        # tokenlizer = model.tokenizer
        # tokenized = tokenlizer(caption)
        # pred_phrases = []
        # for logit, box in zip(logits_filt, boxes_filt):
        #     pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        #     if with_logits:
        #         pred_phrases.append([pred_phrase, logit.max().item()])
        #     else:
        #         pred_phrases.append(pred_phrase)
        boxes_filt, pred_phrases = self.postprocess(logits_filt, boxes_filt, self.ground_model, caption, text_threshold, box_threshold, with_logits=with_logits)
        return boxes_filt, pred_phrases
    

    def postprocess(self, logits_filt, boxes_filt, model, caption, text_threshold, box_threshold, with_logits):
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer, with_location=True)
            if with_logits:
                pred_phrases.append([pred_phrase, logit.max().item()])
            else:
                pred_phrases.append(pred_phrase)
        return boxes_filt, pred_phrases
 
    def ground_caption(self, image_pil, caption):
        image_pil, image = self.load_image(image_pil=image_pil)
        boxes_filt, pred_phrases = self.get_grounding_output(self.ground_model, image, caption, self.box_threshold, self.text_threshold, cpu_only=self.cpu_only)
        return boxes_filt, pred_phrases

    def ground_caption_raw(self, image_pil, caption):
        image_pil, image = self.load_image(image_pil=image_pil)
        logits, boxes = self.get_grounding_output(self.ground_model, image, caption, self.box_threshold, self.text_threshold, cpu_only=self.cpu_only, return_raw=True)
        return logits, boxes


if __name__=="__main__":
	grounder = caption_grounder(args)
	grounder.ground_caption()

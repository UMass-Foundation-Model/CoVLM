import spacy
import torch
from tqdm import tqdm
import numpy as np
import itertools
from PIL import Image
nlp = spacy.load('en_core_web_md')


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

    outputs = outputs[:, len(input_ids[0]):]
    return outputs

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


def prepare_batch_images(batch, image_processor):
    batch_images = None
    for b in batch:
        b_image = image_processor(b["image"]).unsqueeze(0).unsqueeze(1).unsqueeze(0)
        if batch_images is None:
            batch_images = b_image
        else:
            batch_images = torch.cat([batch_images, b_image], dim=0)
    return batch_images


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


def get_iou(box1, box2):
    # box1 and box2 should be in the format [x1, y1, x2, y2]
    intersection = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * \
                   max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection
    iou = intersection / union if union > 0 else 0
    return iou


def find_root(token):
    if token.pos_ == "VERB":
        return token
    while token.dep_ in ["compound", "amod"]:
        token = token.head
    return token

def get_object_from_text(text, verbose=False):
    if len(text.split(" ")) == 3:
        text = text.split(" ")
        return [text[0], text[-1]]
    doc = nlp(text)
    if verbose:
        for TT in doc:
            print(TT.text, TT.pos_, TT.dep_, TT.head)
    roots = set()
    for i, token in enumerate(doc):
        roots.add(find_root(token))
    exprs = []
    roots = sorted(list(roots), key=lambda token: token.idx)
    first_nsubj = True
    if verbose:
        print(roots)
    for root in roots:
        if root.pos_ not in ["NOUN", "PROPN"]:
            continue
        if root.dep_ not in ["pobj", "nsubj"]:
            continue
        if not first_nsubj and root.dep_ in ["nsubj"]:
            continue
        exprs.append([])
        for token in doc:
            if find_root(token) == root:
                exprs[-1].append(token.text)
        exprs[-1] = " ".join(exprs[-1]).replace(" '", "'")
        if exprs[-1] not in text:
            if verbose:
                print("not in text error:", exprs[-1], "#",text)
            # for TT in doc:
            #     print(TT.text, TT.pos_, TT.dep_, TT.head)
            # import pdb; pdb.set_trace()
            exprs.pop()
        if first_nsubj and root.dep_ in ["nsubj"]:
            first_nsubj = False
    if len(exprs) <= 1:
        if verbose:
            print("not enough exprs error:", exprs, "#",text)
        return []
    return exprs


def is_correct(input_ids, logits, tokenizer, object: str, topk=5, N=10):
    answer_id = torch.tensor(tokenizer(f" {object}", add_special_tokens=False)["input_ids"]).to(input_ids.device)
    answer_begin_idx = (input_ids == answer_id[0]).nonzero()
    answer_idx = None
    for (batch_idx, IDX) in answer_begin_idx:
        try:
            if (input_ids[batch_idx, IDX:IDX+len(answer_id)] == answer_id).all():
                answer_idx = list(range(IDX-1, IDX+len(answer_id)-1))
        except:
            pass
    if answer_idx is None:
        return np.inf, False, False
    res = logits[0, answer_idx].softmax(-1).sort(descending=True)
    values = res.values
    indices = res.indices
    chosen_ids = list(itertools.product(*([list(range(N))]*len(answer_idx))))
    probs = []
    for ids in chosen_ids:
        prob = 1.0
        for i, id in enumerate(ids):
            prob *= values[i, id]
        probs.append((prob.item(), ids))
    probs.sort(reverse=True)
    answer_pos = tuple([id_array.tolist().index(idx) for id_array, idx in zip(indices, answer_id)])
    ranking = [p[1] for p in probs]
    # if len(answer_idx) > 1:
    #     import pdb; pdb.set_trace()
    try:
        r = ranking.index(answer_pos)
        return r, r < 1, r < 5
    except:
        return np.inf, False, False


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


def _eval_text_image(text, image, model, tokenizer, image_processor, vis_embed_size, media_token_id, prebox_token_id, debug=False, objects=None):
    batch_images = image_processor(image).unsqueeze(0).unsqueeze(1).unsqueeze(0)
    if objects is None:
        objects = get_object_from_text(text)
    if len(objects) == 0:
        return None, None, None
    if debug:
        tqdm.write(text)
        tqdm.write(f"{objects}")
    first_idx = text.find(objects[0])
    if first_idx == 0:
        first_text = f"<|#object#|>{objects[0]}<|#endofobject#|><|#visual#|>"
    else:
        first_text = text[:first_idx-1] + f"<|#object#|> {objects[0]}<|#endofobject#|><|#visual#|>"
    
    if debug:
        tqdm.write(first_text)
    prompt = [f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>{first_text}"]
    # import pdb; pdb.set_trace()
    # print("do first get_bbox |", first_text)
    first_box, first_score = get_bbox(None, batch_images, prompt, model, tokenizer, media_token_id, prebox_token_id, return_all=False)
    if not model.valid and debug:
        import pdb; pdb.set_trace()
    if first_box is not None:
        added_bbox_list = [torch.tensor(first_box).unsqueeze(0).cuda() / 224]
        text = first_text + "<|#box#|><|#endofobject#|>" + text[first_idx+len(objects[0]):]
    else:
        added_bbox_list = []

    final_ranks = []
    is_top1_list = []
    is_top5_list = []
    for kk, object in enumerate(objects):
        if kk == 0:
            continue
        idx = text.find(objects[0])
        for t_i, temp in enumerate(objects[1:kk+1]):
            # t_i is actually the previous one. This is not a bug
            idx = text.find(temp, idx + len(objects[t_i]))
            while idx+len(temp) != len(text) and (text[idx-1] == "#" or text[idx+len(temp)] == "#"):
                # in case temp is box or object or visual or something like that
                idx = text.find(temp, idx + len(temp))
        this_text = text[:idx-1] + "<|#object#|><|#previsual#|>"
        # if this_text == "<|#object#|><|#previsual#|>":
        #     import pdb; pdb.set_trace()
        if debug:
            tqdm.write(this_text)
        prompt = [f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>{this_text}"]
        # import pdb; pdb.set_trace()
        # print("do pre get_bbox |", this_text)
        pre_boxes, pre_scores = get_bbox(added_bbox_list, batch_images, prompt, model, tokenizer, media_token_id, 
        prebox_token_id, return_all=True)
        if not model.valid and debug:
            import pdb; pdb.set_trace()
        logits_list = []
        # pre_boxes = [pre_boxes[0]]
        # pre_scores = [pre_scores[0]]
        this_text = this_text + f"<|#prebox#|><|#object#|> {object}<|#endofobject#|>"
        for pre_box, pre_score in zip(pre_boxes, pre_scores):
            prompt = [f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>{this_text}"]
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
            this_added_bbox_list = added_bbox_list + [torch.tensor(pre_box).unsqueeze(0).cuda() / 224]

            with torch.cuda.amp.autocast(dtype=torch.float16) and torch.no_grad():
                outputs = model(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask,
                    image_nums=image_nums,
                    image_start_index_list=image_start_index_list,
                    added_bbox_list=this_added_bbox_list,
                    add_box=this_added_bbox_list is not None and len(this_added_bbox_list) != 0,
                    relations=None,
                )
            if not model.valid and debug:
                import pdb; pdb.set_trace()
            logits_list.append([pre_score, outputs.logits])
            if debug:
                answer_start_idx = (lang_x == tokenizer("<|#object#|>", add_special_tokens=False)["input_ids"][-1]).nonzero()[-1][1]
                logits = outputs["logits"][0, answer_start_idx:]
                tqdm.write(tokenizer.decode(logits[0].sort(descending=True).indices.tolist()[:10]))
            # if debug:
            #     image.save("Atest.png")
            #     open_cv_image = np.array(image)
            #     open_cv_image = open_cv_image[:, :, ::-1].copy()
            #     if first_box is not None:
            #         open_cv_image = cv2.rectangle(open_cv_image, first_box[:2].astype(int), first_box[2:].astype(int), (255, 0, 0), 2)
            #     if pre_box is not None:
            #         open_cv_image = cv2.rectangle(open_cv_image, pre_box[:2].astype(int), pre_box[2:].astype(int), (0, 255, 0), 2)
            #     cv2.imwrite(f"Atest.png", open_cv_image)
            #     import pdb; pdb.set_trace()
        pre_scores = np.array([x[0] for x in logits_list])
        final_probs = 0.0
        for score, (_, logits) in zip(pre_scores, logits_list):
            final_probs += score * logits.softmax(-1)
        assert input_ids.shape[:2] == final_probs.shape[:2]
        _rank, is_top1, is_top5 = is_correct(input_ids, final_probs, tokenizer, object, topk=5)
        final_ranks.append(_rank)
        is_top1_list.append(is_top1)
        is_top5_list.append(is_top5)
        this_text = text[:idx-1] + f"<|#object#|> {object}<|#endofobject#|><|#visual#|>"
        if debug:
            tqdm.write(this_text)
        prompt = [f"{tokenizer.bos_token}<|#image#|>{tokenizer.pad_token*vis_embed_size}<|#endofimage#|>{this_text}"]
        # print("do this get_bbox |", this_text)
        this_box, this_score = get_bbox(added_bbox_list, batch_images, prompt, model, tokenizer, media_token_id, prebox_token_id, return_all=False)
        if not model.valid and debug:
            import pdb; pdb.set_trace()
        if this_box is not None:
            added_bbox_list += [torch.tensor(this_box).unsqueeze(0).cuda() / 224]
            text = this_text + "<|#box#|><|#endofobject#|>" + text[idx+len(object):]
    return final_ranks, is_top1_list, is_top5_list


if __name__ == "__main__":
    # print(get_object_from_text("there is a cookie. there is a bear. white orio cookie is next to the teddy bear. car runs on the traffic road. there is a tree.", verbose=False))
    print(get_object_from_text("President speaks to an American at a business office",verbose=True))

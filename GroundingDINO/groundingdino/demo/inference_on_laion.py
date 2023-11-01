import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy, copy
import random
import logging
import spacy
nlp = spacy.load("en_core_web_sm")
# OBJ_LENGTHS = []

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def insert_object(
    caption, object, quant_boxes, object_id,
):
    visual_token = "<|#visual#|>"
    object_token = "<|#object#|>"
    visual_token_with_id = visual_token + f"_{str(object_id).zfill(4)}"
    _caption = caption.lower()
    # the input object should be all lower case, but to be safe, lower it again
    object, loc_idx = object
    _object = object.lower()
    idx = -1
    for i in range(len(_caption)):
        if _caption[i:i+len(_object)] == _object:
            idx += 1
            if loc_idx == idx:
                caption = f"{caption[:i]}{object_token}{caption[i:i+len(_object)]}{visual_token_with_id}{caption[i+len(_object):]}"
                id = int(caption[i+len(object_token)+len(_object):i+len(object_token)+len(_object)+len(visual_token_with_id)].split("_")[-1])
                return (caption, id, quant_boxes)
    return (caption, None, None)


def add_loc_to_text(
    boxes_filt, pred_phrases, caption: str,
    expand=True, delete_contained=False, delete_contained_and_same_object=False,
    always_expand=False,
):
    # assert expand and delete_contained
    unique_objects = {}
    decide_to_expand_object = []
    for (object, score), bbox in zip(pred_phrases, boxes_filt):
        expand_object = None
        if expand and (random.random() < 0.5 or always_expand or (object in decide_to_expand_object)):
            expand_object = expand_expression(caption, object)
            decide_to_expand_object.append(object)
        else:
            if object not in unique_objects:
                unique_objects[object] = []
            unique_objects[object].append([score, bbox])
        if expand_object is not None:
            object = expand_object
            if object not in unique_objects:
                unique_objects[object] = []
            unique_objects[object].append([score, bbox])
    if delete_contained:
        retain_object = []
        all_object = sorted(list(unique_objects.keys()), reverse=True)
        for i, obj in enumerate(all_object):
            retain = True
            for j in range(i):
                if obj[0] in all_object[j][0]:
                    retain = False
                    break
            if retain:
                retain_object.append(obj)
        retain_unique_objects = {}
        for obj in retain_object:
            retain_unique_objects[obj] = deepcopy(unique_objects[obj])
        unique_objects = retain_unique_objects


    if delete_contained_and_same_object:
        retain_object = []
        all_object = sorted(list(unique_objects.keys()), reverse=True)
        for i, obj in enumerate(all_object):
            retain = True
            for j in range(i):
                if obj[0] in all_object[j][0] and calculate_iou(unique_objects[obj][0][1], unique_objects[all_object[j]][0][1]) >= 0.8:
                    retain = False
                    break
            if retain:
                retain_object.append(obj)
        retain_unique_objects = {}
        for obj in retain_object:
            retain_unique_objects[obj] = deepcopy(unique_objects[obj])
        unique_objects = retain_unique_objects

    for obj in unique_objects:
        unique_objects[obj] = sorted(unique_objects[obj], key=lambda x: x[0], reverse=True)
    objects = unique_objects.keys()
    objects = sorted(objects, key=lambda x: len(x[0]), reverse=False)
    added_bbox = {}
    for object_id, object in enumerate(objects):
        if (
            object[0].lower() in "answer the question using a single word or phrase" or 
            object[0].lower() in "answer with the option’s letter from the given choices directly" or
            object[0].lower() in "provide a one-sentence caption for the provided image" or 
            object[0].lower() in "provide a short description for this region" or
            object[0].lower() in "provide the bounding box coordinate of the region this sentence describes" or
            object[0].lower().strip() == "yes" or
            object[0].lower().strip() == "no"
        ):
            continue
        quant_boxes = []
        for score, box in unique_objects[object]:
            box = box.clone()
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            box = torch.clamp(box, 0.0, 1.0)
            quant_boxes.append(box.clone())
        if object[-1] == -1 or len(object[0]) == 0 or len(quant_boxes) == 0:
            continue
        caption, object_id, quant_boxes = insert_object(caption, object, quant_boxes, object_id)
        if object_id is None and quant_boxes is None:
            continue
        # DEBUG
        # OBJ_LENGTHS.append(len(object[0].split(" ")))
        # DEBUG
        if object_id not in added_bbox:
            added_bbox[object_id] = []
        else:
            logging.info(f"{caption}")
        added_bbox[object_id].extend(quant_boxes)
    for id in added_bbox:
        idx = caption.find(f"<|#visual#|>_{str(id).zfill(4)}")
        added_bbox[id] = [idx, added_bbox[id]]
    idx_bbox_pairs = list(added_bbox.values())
    idx_bbox_pairs = sorted(idx_bbox_pairs, key=lambda x: x[0])
    for id in added_bbox:
        caption = caption.replace(f"<|#visual#|>_{str(id).zfill(4)}", "<|#visual#|>")
    bboxes = [torch.vstack(x[1]) for x in idx_bbox_pairs]
    return caption, bboxes


def nms_without_score(boxes, iou_thres = 0.45):
    """ 非极大值抑制 """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2-x1) * (y2-y1)
    keep = []

    # 按置信度进行排序
    index = np.arange(len(boxes))

    while(index.size):
        # 置信度最高的框
        i = index[0]
        keep.append(index[0])

        if(index.size == 1): # 如果只剩一个框，直接返回
            break

        # 计算交集左下角与右上角坐标
        inter_x1 = np.maximum(x1[i], x1[index[1:]])
        inter_y1 = np.maximum(y1[i], y1[index[1:]])
        inter_x2 = np.minimum(x2[i], x2[index[1:]])
        inter_y2 = np.minimum(y2[i], y2[index[1:]])
        # 计算交集的面积
        inter_area = np.maximum(inter_x2-inter_x1, 0) * np.maximum(inter_y2-inter_y1, 0)
        # 计算当前框与其余框的iou
        iou = inter_area / (areas[index[1:]] + areas[i] - inter_area)
        ids = np.where(iou < iou_thres)[0]
        index = index[ids+1]

    return boxes[keep]



def calculate_iou(box1, box2):
    # Box format: [x_min, y_min, x_max, y_max]
    
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of both bounding boxes
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the IoU
    iou = intersection_area / float(area_box1 + area_box2 - intersection_area)

    return iou


# def expand_expression(caption, object):
#     doc = nlp(caption)
#     print(caption)
#     object, III = object
#     start_idx = caption.find(object)
#     while III != 0:
#         start_idx = caption.find(object, start_idx+1)
#         III -= 1
#     if start_idx == -1:
#         return None
#     end_idx = start_idx + len(object) - 1
#     print(start_idx, end_idx)
#     expand_expr = []
#     for token in doc:
#         print(f"{token.idx} Token: {token.text}, Dependency Relation: {token.dep_}, Head: {token.head.text}")
#     for token in doc:
#         root = token
#         while root.head != root and root.dep_ not in ["conj", "cc", "nsubj"]:
#             root = root.head
#         if start_idx <= root.idx and root.idx <= end_idx:
#             expand_expr.append(token)
#     print(expand_expr)
#     if len(expand_expr) == 0:
#         return None 
#     expr_start_idx = min([t.idx for t in expand_expr])
#     expr_end_idx = max([t.idx + len(t.text) - 1 for t in expand_expr])
#     expr = caption[expr_start_idx: expr_end_idx+1]
#     return (expr, III)

def expand_expression(caption, object):
    doc = nlp(caption)
    token_head = {}
    for token in doc:
        id = (token.idx, token.text)
        token_head[id] = []
    for token in doc:
        if token.dep_ in ["conj", "cc", "nsubj"]:
            continue
        parent = token.head
        parent_id = (parent.idx, parent.text)
        token_head[parent_id].append(token)
    object, III = object
    start_idx = caption.find(object)
    ori_III = copy(III)
    while III != 0 and start_idx != -1:
        start_idx = caption.find(object, start_idx+1)
        III -= 1
    if start_idx == -1:
        return None
    end_idx = start_idx + len(object) - 1
    expand_expr = []
    for token in doc:
        if start_idx <= token.idx and token.idx <= end_idx:
            queue = [token]
            visited = []
            while len(queue) != 0:
                token = queue.pop()
                expand_expr.append(token)
                visited.append(token)
                for token in token_head[(token.idx, token.text)]:
                    if token not in visited:
                        queue.insert(0, token)
    if len(expand_expr) == 0:
        return None 
    expr_start_idx = min([t.idx for t in expand_expr])
    expr_end_idx = max([t.idx + len(t.text) - 1 for t in expand_expr])
    expr = caption[expr_start_idx: expr_end_idx+1]
    return (expr, ori_III)


if __name__ == "__main__":

    # grounder = caption_grounder(
    #     config_file="/gpfs/u/home/LMCG/LMCGljnn/scratch/code/multimodal/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    #     checkpoint_path="/gpfs/u/home/LMCG/LMCGljnn/scratch/code/multimodal/GroundingDINO/checkpoints/groundingdino_swint_ogc.pth",
    #     cpu_only=False,
    # )
    # dataset = wds.WebDataset("/gpfs/u/home/LMCG/LMCGljnn/scratch-shared/junyan/raw/laion2b-dedup-ground/s200/00000.tar").decode("pil").to_tuple("jpg", "txt", "logits.pyd", "boxes.pyd")
    # os.makedirs("images", exist_ok=True)
    # for i, (pil_img, caption, logits_filt, boxes_filt) in enumerate(tqdm(dataset, disable=False)):
    #     # boxes_filt, pred_phrases = grounder.ground_caption(image_pil=pil_img, caption=caption)
    #     boxes_filt, pred_phrases = grounder.postprocess(logits_filt, boxes_filt, grounder.ground_model, caption, grounder.text_threshold, grounder.box_threshold, with_logits=True)
    #     caption_with_loc = add_loc_to_text(boxes_filt, pred_phrases, caption)
    #     # print("="*80)
    #     # print(pred_phrases)
    #     tqdm.write(f"{caption_with_loc}")
    #     # print("="*80)
    #     # pil_img.save(f"images/{i}.png")
    # sentence = "the man riding on a horse and the horse which runs on the road next to the man are both visible to the white and rich second man from the left"
    # sentence = "This black man is looking on the street"
    sentence = "a truck drives on a country road"
    print(expand_expression(sentence, ("truck", 0)))

import torch
from einops import rearrange
from torch import nn
from yolox.models.yolo_head import YOLOXHead
from yolox.utils.boxes import xyxy2cxcywh, cxcywh2xyxy
from yolox.utils.demo_utils import nms
import numpy as np
import logging
from transformers import LogitsProcessorList
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %I:%M:%S',
)


class DetectionHead(nn.Module):
    def __init__(self, strides, in_channels, out_channels=4096) -> None:
        super().__init__()
        self.adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.yolox_head = YOLOXHead(
            num_classes=1,
            strides=[strides],
            in_channels=[out_channels],
        )

    def forward(self, xin, labels):
        xin = [self.adapter(xin[0])]
        return self.yolox_head(xin=xin, labels=labels)


class Flamingo(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        image_end_token_id: int,
        visual_token_id: int,
        previsual_token_id: int,
        box_token_id: int,
        prebox_token_id: int,
        endofobject_token_id: int,
        vis_dim: int,
        vis_embed_size: int,
        lang_dim: int,
        hidden_state_dim: int,
        image_size: int,
        patch_size: int,
        mm_projector: torch.nn.Module = None,
    ):
        super().__init__()
        self.image_end_token_id = image_end_token_id
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.box_token_id = box_token_id
        self.prebox_token_id = prebox_token_id
        self.media_token_id = media_token_id
        self.visual_token_id = visual_token_id
        self.previsual_token_id = previsual_token_id
        self.endofobject_token_id = endofobject_token_id
        self.hidden_state_dim = hidden_state_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vis_dim = vis_dim
        self.lang_dim = lang_dim
        self.vis_proj = nn.Linear(self.vis_dim, self.lang_dim) if mm_projector is None else mm_projector
        self.vision_encoder = vision_encoder
        self.num_positions = vis_embed_size
        self.lang_encoder = lang_encoder
        self.lang_encoder.init_flamingo()
        first_layer = self.lang_encoder._get_decoder_layers()[0]
        first_layer.visual_token_id = visual_token_id
        first_layer.media_token_id = media_token_id
        first_layer.box_token_id = box_token_id
        if mm_projector is not None:
            self.detection_head = DetectionHead(patch_size, self.hidden_state_dim + self.lang_dim)
        else:
            self.detection_head = YOLOXHead(
                num_classes=1,
                strides=[patch_size],
                in_channels=[self.hidden_state_dim + self.lang_dim],
            )

    def _get_detection_batch(
        self,
        visual_token_id,
        previsual_token_id,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        added_bbox_list,
        box_num: int = 100,
    ):
        select_mask = torch.logical_or(input_ids == visual_token_id, input_ids == previsual_token_id)
        visual_token_position = select_mask.nonzero()
        visual_token_hidden_states = hidden_states[select_mask]
        prev_batch_idx = -1
        media_idx = []
        cnt = 0
        assert len(visual_token_hidden_states) == len(visual_token_position)
        if len(added_bbox_list) != len(visual_token_position):
            msg = f"ERROR: {len(added_bbox_list)}:{len(visual_token_position)}\n{added_bbox_list}\n{visual_token_position}"
            logging.info(msg)
            alpha = 0.0
        else:
            alpha = 1.0
        visual_batches = []
        previsual_batches = []
        for (batch_idx, idx), visual_token_hidden_state, bbox in zip(
            visual_token_position, visual_token_hidden_states, added_bbox_list,
        ):
            # ! VERY IMPORTANT !
            bbox = bbox.clone()
            # ! VERY IMPORTANT !
            batch_idx = batch_idx.item()
            idx = idx.item()
            if batch_idx != prev_batch_idx:
                prev_batch_idx = batch_idx
                this_input_ids = input_ids[batch_idx]
                cnt += len(media_idx)
                media_idx = (this_input_ids == self.media_token_id).nonzero().reshape(-1).tolist()
            for i in range(len(media_idx)):
                if i == len(media_idx) - 1 or idx > media_idx[i] and idx < media_idx[i+1]:
                    break
            image_index = cnt + i
            size = int(self.image_embedding[image_index].shape[0] ** 0.5)
            image_embedding = self.image_embedding[image_index]
            bbox = xyxy2cxcywh(bbox) * self.image_size
            # print(bbox)
            concat_image_visual_embedding = torch.cat([image_embedding, visual_token_hidden_state.unsqueeze(0).repeat(image_embedding.shape[0], 1)], dim=-1).reshape(size, size, -1)
            label = torch.cat([torch.zeros(bbox.shape[0], 1, device=bbox.device), bbox], dim=-1)
            label = torch.cat([label, torch.zeros(box_num - label.shape[0], label.shape[1], device=label.device)], dim=0)
            if input_ids[batch_idx, idx] == previsual_token_id:
                previsual_batches.append([concat_image_visual_embedding, label])
            elif input_ids[batch_idx, idx] == visual_token_id:
                visual_batches.append([concat_image_visual_embedding, label])
            else:
                logging.info(f"WARNING... NOT visual nor previsual. it is {input_ids[batch_idx, idx]}")
        return visual_batches, previsual_batches, alpha, alpha

    def get_detection_losses(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        added_bbox_list,
        box_num: int = 100,
    ):
        visual_token_batches, previsual_token_batches, alpha1, alpha2 = self._get_detection_batch(
            visual_token_id=self.visual_token_id,
            previsual_token_id=self.previsual_token_id,
            input_ids=input_ids,
            hidden_states=hidden_states,
            added_bbox_list=added_bbox_list,
            box_num=box_num,
        )
        loss_dict = []
        for batches, alpha in zip([visual_token_batches, previsual_token_batches], [alpha1, alpha2]):
            # x: [B, C, H, W]
            if len(batches) != 0:
                x = torch.cat([batch[0].unsqueeze(0) for batch in batches], dim=0).permute(0,3,1,2)
                labels = torch.cat([batch[1].unsqueeze(0) for batch in batches], dim=0)
            else:
                x = None
                labels = None
            if x is not None:
                losses = self.detection_head(xin=[x], labels=labels)
                loss, loss_iou, loss_obj, loss_cls, loss_l1, _ = losses
            else:
                loss = torch.tensor(0.0).cuda()
                loss_iou = loss
                loss_obj = loss
                loss_cls = loss
                loss_l1 = loss

            loss_dict.append(dict(
                loss=loss * alpha,
                loss_iou=loss_iou * alpha,
                loss_obj=loss_obj * alpha,
                loss_cls=loss_cls * alpha,
                loss_l1=loss_l1 * alpha,
            ))
        ret_loss = {}
        for key in loss_dict[0].keys():
            ret_loss[key] = 0.0
            for d in loss_dict:
                ret_loss[key] += d[key]
        return ret_loss, loss_dict

    def get_detection_result(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        nms_thr: float = 0.45,
        score_thr: float = 0.01,
    ):
        assert len(input_ids) == 1, "only batch size = 1 is supported yet"
        visual_token_hidden_state = hidden_states[..., -1, :]
        boxes_list = []
        scores_list = []
        for image_embedding in self.image_embedding:
            size = int(image_embedding.shape[0] ** 0.5)
            x = torch.cat([
                image_embedding.to(visual_token_hidden_state.device),
                visual_token_hidden_state.repeat(image_embedding.shape[0], 1),
            ], dim=-1).reshape(size, size, -1).unsqueeze(0).permute(0, 3, 1, 2)
            with torch.no_grad():
                outputs = self.detection_head(xin=[x], labels=None)
            boxes = outputs[0, :, :4].cpu().numpy()
            scores = outputs[0, :, 4].cpu().numpy()
            scores_mask = scores > score_thr
            boxes = boxes[scores_mask]
            boxes = cxcywh2xyxy(boxes)
            scores = scores[scores_mask]
            keep = nms(boxes, scores, nms_thr=nms_thr)
            boxes = boxes[keep]
            scores = scores[keep]
            boxes_list.append(boxes)
            scores_list.append(scores)
        if len(boxes_list) == 1:
            boxes_list = boxes_list[0]
            scores_list = scores_list[0]
        return boxes_list, scores_list

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cached_vision_x: bool = False,
        clear_conditioned_layers: bool = True,
        past_key_values: torch.Tensor = None,
        use_cache: bool = False,
        image_nums: int = None,
        image_start_index_list: list = None,
        added_bbox_list: list = None,
    ):
        self.valid = True
        self.lang_encoder.loc_list = None
        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()
        else:
            if sum(image_nums) != 0:
                # Case: do not use caching (i.e. this is a standard forward pass);
                self._encode_vision_x(
                    vision_x=vision_x,
                    image_nums=image_nums,
                    image_start_index_list=image_start_index_list,
                    added_bbox_list=added_bbox_list,
                    input_ids=lang_x,
                )
        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
        )
        hidden_states = output["hidden_states"][-1]
        if self.training and added_bbox_list is not None:
            detection_losses, loss_dict = self.get_detection_losses(
                input_ids=lang_x,
                hidden_states=hidden_states,
                added_bbox_list=added_bbox_list,
            )
            output["detection_losses"] = detection_losses
            output["loss_dict"] = loss_dict
        elif labels is None:
            boxes, scores = self.get_detection_result(
                input_ids=lang_x,
                hidden_states=hidden_states,
            )
            output["boxes"] = boxes
            output["scores"] = scores

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()
        return output

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        added_bbox_list=None,
        num_beams=1,
        max_new_tokens=None,
        min_new_tokens=None,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        no_repeat_ngram_size=0,
        prefix_allowed_tokens_fn=None,
        length_penalty=1.0,
        num_return_sequences=1,
        do_sample=False,
        early_stopping=False,
        bad_words_ids=None,
        force_words_ids=None,
        image_start_index_list=None,
        image_nums=None,
        min_length=None,
        return_dict_in_generate=False,
        output_hidden_states=False,
        output_scores=False,
        logits_processor_list=None,
        eos_token_id=None,
    ):
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)
            image_start_index_list = torch.tensor(image_start_index_list).repeat_interleave(num_beams, dim=0).tolist()
            image_nums = torch.tensor(image_nums).repeat_interleave(num_beams, dim=0).tolist()
            if added_bbox_list is not None and len(added_bbox_list) != 0:
                added_bbox_list = added_bbox_list * num_beams

        self._encode_vision_x(vision_x=vision_x, image_nums=image_nums, image_start_index_list=image_start_index_list, num_beams=num_beams, added_bbox_list=added_bbox_list, input_ids=lang_x.repeat_interleave(num_beams, dim=0))

        if logits_processor_list is not None:
            assert isinstance(logits_processor_list, list)
            logits_processor_list = LogitsProcessorList(logits_processor_list)
        output = self.lang_encoder.generate(
            input_ids=lang_x,
            attention_mask=attention_mask,
            eos_token_id=(self.eoc_token_id) if eos_token_id is None else eos_token_id,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            min_new_tokens=min_new_tokens,
            length_penalty=length_penalty,
            logits_processor=logits_processor_list,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
        )
        self.lang_encoder.clear_conditioned_layers()
        return output

    def _get_data_list_and_visual_tokens(
        self,
        all_box_list,
        box_token_id,
        prebox_token_id,
        input_ids,
        vision_x,
    ):
        box_locations = (torch.logical_or(input_ids == box_token_id, input_ids == prebox_token_id)).nonzero()
        prev_batch_idx = -1
        media_idx = []
        cnt = 0
        data_list = []
        visual_tokens = []
        if len(all_box_list) == len(box_locations) * 4:
            all_box_list = np.array(all_box_list).reshape(-1, 4).tolist()
        if len(all_box_list) != len(box_locations):
            logging.info(f"WARNING. len(all_box_list) != len(box_locations) {len(all_box_list)} vs {len(box_locations)}")
            self.valid = False
        for III, (batch_idx, idx) in enumerate(box_locations):
            batch_idx = batch_idx.item()
            idx = idx.item()
            if batch_idx != prev_batch_idx:
                prev_batch_idx = batch_idx
                this_input_ids = input_ids[batch_idx]
                cnt += len(media_idx)
                media_idx = (this_input_ids == self.media_token_id).nonzero().reshape(-1).tolist()
            for i in range(len(media_idx)):
                if i == len(media_idx) - 1 or idx > media_idx[i] and idx < media_idx[i + 1]:
                    break
            image_index = cnt + i
            size = int(vision_x[image_index].shape[0] ** 0.5)
            image_feature = vision_x[image_index].reshape(size, size, -1)
            try:
                raw_xyxy = all_box_list[III]
            except IndexError:
                logging.info("out of scope for all_box_list")
                raw_xyxy = all_box_list[-1]
            region_xyxy = np.array(raw_xyxy) * size
            x1, y1, x2, y2 = region_xyxy.astype(int).clip(0, size - 1).tolist()
            x2 = max(x1, x2)
            y2 = max(y1, y2)
            visual_token = image_feature[y1: y2 + 1, x1: x2 + 1].reshape(-1, image_feature.shape[-1]).mean(0)
            box = torch.tensor([0] + raw_xyxy, device=visual_token.device, dtype=visual_token.dtype)
            data_list.append([visual_token, box, batch_idx, idx, i])
            visual_tokens.append(visual_token)
        return data_list, visual_tokens

    def _encode_vision_x(self, vision_x: torch.Tensor, image_nums=None, image_start_index_list=None, added_bbox_list=None, num_beams=None, input_ids=None):
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        if hasattr(self.vision_encoder, "visual"):
            vision_x = self.vision_encoder.visual(vision_x)[1]
        else:
            vision_x = self.vision_encoder(vision_x).flatten(2)
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        vision_x = vision_x.mean(2)
        vision_x = self.vis_proj(vision_x).squeeze(1)
        self.image_embedding = vision_x

        data_list = None
        visual_tokens = None
        if added_bbox_list is not None and input_ids is not None:
            all_box_list = added_bbox_list[0].tolist()
            for list in added_bbox_list[1:]:
                all_box_list.extend(list.tolist())
            data_list, visual_tokens = self._get_data_list_and_visual_tokens(
                all_box_list=all_box_list,
                box_token_id=self.box_token_id,
                prebox_token_id=self.prebox_token_id,
                input_ids=input_ids,
                vision_x=vision_x,
            )

        first_layer = self.lang_encoder._get_decoder_layers()[0]
        first_layer.condition_vis_x(vision_x, image_nums, image_start_index_list, num_beams=num_beams, visual_tokens=visual_tokens, data_list=[[d[2], d[3]] for d in data_list] if data_list is not None else data_list)

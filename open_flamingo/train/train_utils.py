import time
from contextlib import suppress
import numpy as np

import torch
from tqdm import tqdm
import datetime
import os
import gc
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from torch.utils.tensorboard import SummaryWriter
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %I:%M:%S',
)


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp_fp16":
        return lambda: torch.cuda.amp.autocast(dtype=torch.float16)
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


def get_sync(model, flag):
    if flag:
        return suppress
    else:
        return lambda: model.no_sync()


def train_one_epoch(
    args,
    model,
    laion_loader,
    pile_loader,
    extra_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    writer: SummaryWriter,
    scaler,
    total_laion_token: int,
    total_pile_token: int,
    total_laion_sample: int,
    total_step: int,
):
    world_size = torch.distributed.get_world_size()
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<|#image#|>", add_special_tokens=False)["input_ids"][-1]
    endofmedia_token_id = tokenizer("<|#endofimage#|>", add_special_tokens=False)["input_ids"][-1]
    visual_token_id = tokenizer("<|#visual#|>", add_special_tokens=False)["input_ids"][-1]
    box_token_id = tokenizer("<|#box#|>", add_special_tokens=False)["input_ids"][-1]
    endofobject_token_id = tokenizer("<|#endofobject#|>", add_special_tokens=False)["input_ids"][-1]
    endofattr_token_id = tokenizer("<|#endofattr#|>", add_special_tokens=False)["input_ids"][-1]
    prebox_token_id = tokenizer("<|#prebox#|>", add_special_tokens=False)["input_ids"][-1]
    previsual_token_id = tokenizer("<|#previsual#|>", add_special_tokens=False)["input_ids"][-1]
    if args.rank == 0:
        logging.info(f"train from: {total_step} step")
    model.train()
    # loop through dataloader
    last_logging_step = total_step
    last_save_step = total_step
    logging.info("start training")
    pile_loader = iter(pile_loader)
    for num_steps, (batch_laion, batch_extra) in tqdm(
        enumerate(zip(laion_loader, extra_loader)),
        disable=args.rank != 0 or "SLURM_PROCID" in os.environ,
        total=args.num_steps * args.gradient_accumulation_steps,
        initial=total_step * args.gradient_accumulation_steps,
    ):
        images = (
            batch_laion[0]
            .to(device_id, dtype=cast_dtype, non_blocking=True)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        image_nums = batch_laion[1]
        image_start_index_list = batch_laion[2]

        input_ids = batch_laion[3].to(device_id, non_blocking=True).long()
        attention_mask = batch_laion[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        added_bbox_list = [x.to(device_id) for x in batch_laion[5]] # list object
        total_laion_token += int(attention_mask.sum().long()) * world_size
        total_laion_sample += sum(image_nums) * world_size


        labels = input_ids.clone()
        labels[input_ids == visual_token_id] = -100
        labels[input_ids == box_token_id] = -100
        labels[input_ids == endofattr_token_id] = -100
        labels[input_ids == previsual_token_id] = -100
        labels[input_ids == prebox_token_id] = -100
        labels[torch.roll(input_ids == prebox_token_id, 1)] = -100
        labels[torch.roll(input_ids == box_token_id, 1)] = -100
        labels[:, 0] = -100
        labels[input_ids == tokenizer.pad_token_id] = -100
        labels[input_ids == media_token_id] = -100
        labels[input_ids == endofmedia_token_id] = -100
        labels.to(device_id)
        current_laion_num = input_ids.shape[0]

        if args.pile_freq != 0 and ((num_steps != 0 and num_steps % args.pile_freq == 0) or args.pile_freq == 1):
            batch_pile = next(pile_loader)
            input_ids2 = batch_pile[0].to(device_id, non_blocking=True).long()
            attention_mask2 = batch_pile[1].to(device_id, dtype=cast_dtype, non_blocking=True)
            input_length = input_ids.shape[-1]
            input_ids2 = torch.cat([input_ids2, torch.ones((input_ids2.shape[0], input_length - input_ids2.shape[1]), device=input_ids2.device, dtype=input_ids2.dtype) * tokenizer.pad_token_id], dim=-1)
            attention_mask2 = torch.cat([attention_mask2, torch.zeros((attention_mask2.shape[0], input_length - attention_mask2.shape[1]), device=attention_mask2.device, dtype=attention_mask2.dtype)], dim=-1)
            labels2 = input_ids2.clone()
            labels2[labels2 == tokenizer.pad_token_id] = -100
            labels2[:, 0] = -100
            labels2.to(device_id)

            image_nums = image_nums + [0] * len(input_ids2)
            image_start_index_list = image_start_index_list + [[]] * len(input_ids2)
            input_ids = torch.cat([input_ids, input_ids2], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask2], dim=0)
            labels = torch.cat([labels, labels2], dim=0)
            total_pile_token += int(attention_mask2.sum().long()) * world_size

        if len(added_bbox_list) == 0:
            added_bbox_list = None
        update_flag = (num_steps != 0 and num_steps % args.gradient_accumulation_steps == 0) or args.gradient_accumulation_steps == 1
        with autocast():
            outputs = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                image_nums=image_nums,
                image_start_index_list=image_start_index_list,
                added_bbox_list=added_bbox_list,
            )
            loss_total = outputs.loss.reshape(labels.shape[0], -1)
            loss_sample = loss_total.sum(-1) / (loss_total != 0).sum(-1)
            loss_sample_for_laion = loss_sample[:current_laion_num]
            nan_mask = torch.isnan(loss_sample_for_laion)
            if nan_mask.sum() > 0:
                logging.warning(f"caption NaN: {nan_mask}")
            if nan_mask.sum() == len(loss_sample_for_laion) or not model.valid:
                logging.info("WARNING: skip this caption loss due to some error")
                loss_laion = torch.tensor(0.0).cuda()
            else:
                loss_laion = loss_sample_for_laion[~nan_mask].mean()
            loss_caption = loss_laion
            divided_loss_laion = loss_laion / args.gradient_accumulation_steps
            if current_laion_num != loss_sample.shape[0]:
                loss_pile = loss_sample[current_laion_num:].mean()
            else:
                loss_pile = torch.tensor(0.0).cuda()
            divided_loss_pile = loss_pile / args.gradient_accumulation_steps

            if "detection_losses" in outputs:
                loss_det = outputs["detection_losses"]["loss"]
                loss_iou = outputs["detection_losses"]["loss_iou"]
                loss_obj = outputs["detection_losses"]["loss_obj"]
                loss_cls = outputs["detection_losses"]["loss_cls"]
            else:
                loss_det = torch.tensor(0.0).cuda()
                loss_iou = torch.tensor(0.0).cuda()
                loss_obj = torch.tensor(0.0).cuda()
                loss_cls = torch.tensor(0.0).cuda()

            if "loss_dict" in outputs:
                visual_loss_iou = outputs["loss_dict"][0]["loss_iou"]
                previsual_loss_iou = outputs["loss_dict"][1]["loss_iou"]
                visual_loss_obj = outputs["loss_dict"][0]["loss_obj"]
                previsual_loss_obj = outputs["loss_dict"][1]["loss_obj"]
            else:
                visual_loss_iou = torch.tensor(0.0).cuda()
                previsual_loss_iou = torch.tensor(0.0).cuda()
                visual_loss_obj = torch.tensor(0.0).cuda()
                previsual_loss_obj = torch.tensor(0.0).cuda()

            divided_loss_det = loss_det / args.gradient_accumulation_steps
            loss = (
                divided_loss_laion * args.loss_multiplier_caption +
                divided_loss_pile * args.loss_multiplier_pile +
                divided_loss_det * args.loss_multiplier_det
            )

        scaler.scale(loss).backward()

        # for logging only
        loss = (
            loss_laion * args.loss_multiplier_caption +
            loss_pile * args.loss_multiplier_pile +
            loss_det * args.loss_multiplier_det
        ).detach()

        # step optimizer and log
        if update_flag:
            total_step += 1
            scaler.unscale_(optimizer)
            if args.ddp:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            else:
                model.clip_grad_norm_(1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()
            # https://github.com/facebookresearch/fairscale/issues/627
            model.zero_grad(set_to_none=True)

        if args.rank == 0 and total_step % args.logging_steps == 0 and total_step != last_logging_step:
            last_logging_step = total_step
            global_step = total_step
            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr", lr, global_step)
            writer.add_scalar("scale", scaler.get_scale(), global_step)
            writer.add_scalar("loss_groundcaption", loss_laion.item(), global_step)
            writer.add_scalar("loss_laion", loss_caption.item(), global_step)
            writer.add_scalar("loss_pile", loss_pile.item(), global_step)
            writer.add_scalar("loss", loss.item(), global_step)
            writer.add_scalar("loss_det", loss_det.item(), global_step)
            writer.add_scalar("loss_iou", loss_iou.item(), global_step)
            writer.add_scalar("loss_obj", loss_obj.item(), global_step)
            writer.add_scalar("loss_cls", loss_cls.item(), global_step)
            writer.add_scalar("loss_iou_visual", visual_loss_iou.item(), global_step)
            writer.add_scalar("loss_obj_visual", visual_loss_obj.item(), global_step)
            writer.add_scalar("loss_iou_previsual", previsual_loss_iou.item(), global_step)
            writer.add_scalar("loss_obj_previsual", previsual_loss_obj.item(), global_step)

            global_sample_num = total_laion_sample
            writer.add_scalar("loss_groundcaption_vs_sample_num", loss_laion.item(), global_sample_num)
            writer.add_scalar("loss_laion_vs_sample_num", loss_caption.item(), global_sample_num)
            writer.add_scalar("loss_pile_vs_sample_num", loss_pile.item(), global_sample_num)
            writer.add_scalar("loss_vs_sample_num", loss.item(), global_sample_num)
            writer.add_scalar("loss_det_vs_sample_num", loss_det.item(), global_sample_num)
            writer.add_scalar("loss_iou_vs_sample_num", loss_iou.item(), global_sample_num)
            writer.add_scalar("loss_obj_vs_sample_num", loss_obj.item(), global_sample_num)
            writer.add_scalar("lr_vs_sample_num", optimizer.param_groups[0]["lr"], global_sample_num)

            writer.add_scalar("loss_groundcaption_vs_token", loss_laion.item(), total_laion_token)
            writer.add_scalar("loss_laion_vs_token", loss_caption.item(), total_laion_token)
            writer.add_scalar("loss_pile_vs_token", loss_pile.item(), total_pile_token)
            writer.add_scalar("loss_det_vs_token", loss_det.item(), total_laion_token)
            writer.add_scalar("loss_iou_vs_token", loss_iou.item(), total_laion_token)
            writer.add_scalar("loss_obj_vs_token", loss_obj.item(), total_laion_token)
            writer.add_scalar("loss_cls_vs_token", loss_cls.item(), total_laion_token)
            total_token = total_laion_token + total_pile_token
            writer.add_scalar("sample_num", global_sample_num, global_step)
            writer.add_scalar("total_laion_token", total_laion_token, global_step)
            writer.add_scalar("total_pile_token", total_pile_token, global_step)
            writer.add_scalar("total_token", total_token, global_step)
            logging.info(
                f"[{global_step}][{total_laion_sample}][{total_token}]. total: {loss.item():.3f} //  caption: {loss_caption.item():.3f} // pile: {loss_pile.item():.3f} // iou: {loss_iou.item():.4f} // obj: {loss_obj.item():.4f} // previsual_obj: {previsual_loss_obj.item():.4f} // visual_obj: {visual_loss_obj.item():.4f} // previsual_iou: {previsual_loss_iou.item():.4f} // visual_iou: {visual_loss_iou.item():.4f} // lr: {lr:.2e} // scale: {scaler.get_scale()}"
            )

        if total_step % args.save_interval == 0 and total_step != last_save_step:
            last_save_step = total_step
            torch.distributed.barrier()
            if args.ddp:
                cpu_state = model.state_dict()
            else:
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(
                    model, StateDictType.FULL_STATE_DICT, save_policy
                ):
                    cpu_state = model.state_dict()
                torch.distributed.barrier()
            if args.rank == 0:
                checkpoint_dict = {
                    "model_state_dict": cpu_state,
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "total_pile_token": total_pile_token,
                    "total_laion_token": total_laion_token,
                    "total_laion_sample": total_laion_sample,
                    "total_step": total_step,
                }
                logging.info(f"Saving checkpoint to {args.run_name}/checkpoint_{total_step}.pt")
                torch.save(checkpoint_dict, f"{args.run_name}/checkpoint_{total_step}.pt")
                del checkpoint_dict
                if args.delete_previous_checkpoint and total_step - args.save_interval > 0 and (total_step - args.save_interval) % args.skip_delete_pattern != 0:
                    try:
                        os.remove(f"{args.run_name}/checkpoint_{total_step-args.save_interval}.pt")
                    except FileNotFoundError:
                        pass
            torch.distributed.barrier()

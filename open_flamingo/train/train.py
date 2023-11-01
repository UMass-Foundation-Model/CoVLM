""" Main training script """

import argparse
import copy
import glob
import os
import random
import functools

import numpy as np
import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
import wandb
from dataset import get_data
from distributed import init_distributed_device, world_info_from_env
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    CPUOffload,
    StateDictType,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from train_utils import train_one_epoch
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from open_flamingo import create_model_and_transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.distributed.optim import ZeroRedundancyOptimizer
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %I:%M:%S',
)


class FakeDataloader:
    def __iter__(self):
        return self

    def __next__(self):
        return None


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def get_grouped_params(model, args):
    params_with_wd, params_without_wd = [], []

    def apply_decay(x):
        x = x.lower()
        return "norm" not in x and "bn" not in x and "bias" not in x and "embed" not in x and "wte" not in x and "flat_param" not in x

    for n, p in model.named_parameters():
        # if p.requires_grad:
        if apply_decay(n):
            if torch.distributed.get_rank() == 0:
                logging.info(f"with wd: {n}")
            params_with_wd.append(p)
        else:
            if torch.distributed.get_rank() == 0:
                logging.info(f"without wd: {n}")
            params_without_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": args.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="datacomp_xl_s13b_b90k", type=str)
    parser.add_argument("--lm_path", default="EleutherAI/pythia-1.4b", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="EleutherAI/pythia-1.4b",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="covlm",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_steps", type=int, default=300000)
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="log loss every n steps"
    )
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size_caption", type=int, default=8)
    parser.add_argument("--batch_size_pile", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument(
        "--caption_shards",
        type=str,
        help="path to laion shards, this should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar",
    )
    parser.add_argument(
        "--pile_shards",
        type=str,
        default=None,
        help="path to pile shards, this should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--loss_multiplier_caption", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_pile", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_det", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    # weight decay is only apply to YOLOX head if using FSDP
    # https://medium.com/@huanghaian123/optimize-and-accelerate-yolox-with-rtmdet-hyps-in-mmyolo-80fc06d61159
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_fp16", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset_resampled", action="store_true")
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument(
        "--checkpoint_activations",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--freeze_vision_encoder",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--vis_embed_size",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--save_interval",
        default=1000,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--skip_delete_pattern",
        default=1500,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--ddp",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--pile_freq",
        default=1,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--restart",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--max-length",
        default=608,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--image-size",
        default=256,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--load_detection_head_weight",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    print(f"local_rank: {args.local_rank} rank: {args.rank} world_size: {args.world_size}")
    device_id = init_distributed_device(args)

    random_seed(args.seed)
    model, image_processor, tokenizer, args.vis_embed_size = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        use_local_files=args.offline,
        checkpoint_activations=args.checkpoint_activations,
        freeze_vision_encoder=args.freeze_vision_encoder,
        load_detection_head_weight=args.load_detection_head_weight,
    )
    if args.rank == 0:
        print(args)
        print(image_processor)

    random_seed(args.seed, args.rank)
    device_id = args.rank % torch.cuda.device_count()
    if args.ddp:
        print("use ddp mode")
        model = model.to(device_id)
        model = DDP(model)
    else:
        fpSixteen = MixedPrecision(
            param_dtype=torch.float16,
            # Gradient communication precision.
            reduce_dtype=torch.float16,
            # Buffer precision.
            # don't enable this
            # buffer_dtype=torch.float16,
        )
        from open_clip.transformer import ResidualAttentionBlock
        from open_flamingo.src.flamingo_lm import FlamingoLayer
        from segment_anything.modeling.image_encoder import Block
        transformer_layer_cls = [
            FlamingoLayer,
            ResidualAttentionBlock,
            Block,
        ]
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_cls,
        )
        ignored_modules = [model.detection_head]
        if args.freeze_vision_encoder:
            ignored_modules += [model.vision_encoder]
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=fpSixteen,
            device_id=torch.cuda.current_device(),
            ignored_modules=ignored_modules,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        )
        model = model.to(device_id)

    pile_dataset = None
    if args.pile_shards is not None:
        pile_dataset = get_data(args, image_processor, tokenizer, "pile")
    laion_dataset = get_data(args, image_processor, tokenizer, "ground_image_text")

    optim_groups = get_grouped_params(model, args)
    if args.ddp:
        assert args.optimizer == "adamw"
        optimizer = torch.optim.AdamW(optim_groups, lr=args.learning_rate)
    else:
        if args.optimizer == "adamw":
            print("use adamw")
            optimizer = torch.optim.AdamW(optim_groups, lr=args.learning_rate)
        elif args.optimizer == "sgd":
            print("use sgd...")
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        else:
            raise NotImplementedError

    total_training_steps = args.num_steps

    if args.rank == 0:
        logging.info(f"Total training steps: {total_training_steps}")

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )
    if args.ddp:
        scaler = GradScaler()
    else:
        scaler = ShardedGradScaler()
    total_laion_token = 0
    total_pile_token = 0
    total_laion_sample = 0
    total_step = 0

    # check if a checkpoint exists for this run
    if os.path.exists(f"{args.run_name}"):
        checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            if args.rank == 0:
                logging.info(f"Found no checkpoints for run {args.run_name}.")
        else:
            args.resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            if args.rank == 0:
                logging.info(f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}.")
            args.restart = False
            if args.rank == 0:
                logging.info("do not restart because an existed checkpoint is found")
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            logging.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        torch.distributed.barrier()
        if args.ddp:
            model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
            # sharded_osd = checkpoint['optimizer_state_dict']
        else:
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                model_state_dict = model.state_dict()
                for key in checkpoint["model_state_dict"].keys():
                    if model_state_dict[key].shape != checkpoint["model_state_dict"][key].shape:
                        if args.rank == 0:
                            logging.info(f'{key}: shape mismatched! {model_state_dict[key].shape} vs {checkpoint["model_state_dict"][key].shape}')
                        checkpoint["model_state_dict"][key] = model_state_dict[key].clone()
                del model_state_dict
                model.load_state_dict(checkpoint["model_state_dict"], False)
            # sharded_osd = FSDP.shard_full_optim_state_dict(checkpoint['optimizer_state_dict'], model, optim_input=optim_groups)
        if not args.restart:
            # optimizer.load_state_dict(sharded_osd)
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            # scaler.load_state_dict(checkpoint["scaler_state_dict"])
            total_laion_token = checkpoint.get("total_laion_token", 0)
            total_pile_token = checkpoint.get("total_pile_token", 0)
            total_laion_sample = checkpoint.get("total_laion_sample", 0)
            total_step = checkpoint.get("total_step", 0)
            if args.rank == 0:
                logging.info("load training statistics...")
        else:
            if args.rank == 0:
                logging.info("restart training / finetuning. only load model weight...")
        del checkpoint
        if args.reset_llm:
            del llm_state_dict
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    model.train()
    if args.rank == 0:
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)
        writer = SummaryWriter(log_dir=os.path.join(args.run_name, "tblog"))
    else:
        writer = None

    laion_dataset.set_epoch(total_step)
    laion_loader = laion_dataset.dataloader
    if pile_dataset is not None:
        pile_dataset.set_epoch(total_step)
        pile_loader = pile_dataset.dataloader
    else:
        pile_loader = FakeDataloader()
    extra_loader = FakeDataloader()
    train_one_epoch(
        args=args,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        laion_loader=laion_loader,
        pile_loader=pile_loader,
        extra_loader=extra_loader,
        device_id=device_id,
        writer=writer,
        scaler=scaler,
        total_laion_token=total_laion_token,
        total_pile_token=total_pile_token,
        total_laion_sample=total_laion_sample,
        total_step=total_step,
    )


if __name__ == "__main__":
    main()

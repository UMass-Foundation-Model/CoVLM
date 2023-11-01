import argparse
from collections import defaultdict
import hashlib
import torch
from open_flamingo.src.factory import create_model_and_transforms
from open_flamingo.train.distributed import world_info_from_env
from open_flamingo.eval.task.cola import evaluate_cola
from open_flamingo.eval.task.gqa import evaluate_gqa
from open_flamingo.eval.task.aro import evaluate_aro
from open_flamingo.eval.task.refcoco import evaluate_refcoco
from open_flamingo.eval.task.vqa import evaluate_vqa
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_path", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--lm_tokenizer_path", type=str, default="facebook/opt-30b")
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)

    # Per-dataset evaluation flags
    parser.add_argument(
        "--eval_vqav2",
        action="store_true",
        default=False,
        help="Whether to evaluate on VQAV2.",
    )
    parser.add_argument(
        "--eval_refcoco",
        action="store_true",
        default=False,
        help="Whether to evaluate on RefCOCO.",
    )

    # Dataset arguments
    # COCO Dataset
    parser.add_argument(
        "--coco_image_dir_path",
        type=str,
        help="Path to the flickr30/flickr30k_images directory.",
        default=None,
    )
    parser.add_argument(
        "--coco_annotations_json_path",
        type=str,
        default=None,
    )

    # VQAV2 Dataset
    parser.add_argument(
        "--vqav2_image_dir_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--vqav2_questions_json_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--vqav2_annotations_json_path",
        type=str,
        default=None,
    )
    # RefCOCO dataset
    parser.add_argument("--refcoco_tsvfile", type=str, default=None)
    # distributed training
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
        "--dist",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--id",
        default=0,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--eval_gqa",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--eval_aro",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--eval_cola",
        default=False,
        action="store_true",
    )
    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()
    if args.dist:
        args.local_rank, args.rank, args.world_size = world_info_from_env()
        init_distributed_device(args)
        print(
            f"local_rank: {args.local_rank} rank: {args.rank} world_size: {args.world_size}"
        )
    else:
        args.local_rank = 0
        args.rank = 0
        args.world_size = 1
        print(f"rank: {args.rank} world_size: {args.world_size}")

    args.id = hashlib.sha224(args.checkpoint_path.encode()).hexdigest()

    # load model
    flamingo, image_processor, tokenizer, vis_embed_size = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.lm_tokenizer_path,
    )
    if args.rank == 0:
        print(args)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model_state_dict = {}
    for key in checkpoint["model_state_dict"].keys():
        model_state_dict[key.replace("module.", "")] = checkpoint["model_state_dict"][
            key
        ]
    if "vision_encoder.logit_scale" in model_state_dict:
        # previous checkpoint has some unnecessary weights
        del model_state_dict["vision_encoder.logit_scale"]
        del model_state_dict["vision_encoder.visual.proj"]
        del model_state_dict["vision_encoder.visual.ln_post.weight"]
        del model_state_dict["vision_encoder.visual.ln_post.bias"]
    flamingo.load_state_dict(model_state_dict, strict=True)
    results = defaultdict(list)
    flamingo.to("cuda")
    if args.eval_vqav2:
        print("Evaluating on VQAv2...")
        vqa_score = evaluate_vqa(
            model=flamingo,
            tokenizer=tokenizer,
            image_processor=image_processor,
            batch_size=args.batch_size,
            image_dir_path=args.vqav2_image_dir_path,
            questions_json_path=args.vqav2_questions_json_path,
            annotations_json_path=args.vqav2_annotations_json_path,
            vqa_dataset="vqa",
            vis_embed_size=vis_embed_size,
            rank=args.rank,
            world_size=args.world_size,
            id=args.id,
        )
        results["vqav2"].append({"score": vqa_score})

    if args.eval_gqa:
        print("Evaluating on GQA...")
        gqa_score = evaluate_gqa(
            model=flamingo,
            tokenizer=tokenizer,
            image_processor=image_processor,
            batch_size=args.batch_size,
            vis_embed_size=vis_embed_size,
            rank=args.rank,
            world_size=args.world_size,
            id=args.id,
        )
        results["gqa"].append({"score": gqa_score})

    if args.eval_refcoco:
        print("Evaluating on RefCOCO...")
        refcoco_score = evaluate_refcoco(
            model=flamingo,
            tokenizer=tokenizer,
            image_processor=image_processor,
            tsvfile=args.refcoco_tsvfile,
            vis_embed_size=vis_embed_size,
            rank=args.rank,
            world_size=args.world_size,
            id=args.id,
        )
        results["refcoco"].append({"score": refcoco_score})

    if args.eval_aro:
        print("Evaluating on ARO...")
        aro_score = evaluate_aro(
            model=flamingo,
            tokenizer=tokenizer,
            image_processor=image_processor,
            vis_embed_size=vis_embed_size,
            rank=args.rank,
            world_size=args.world_size,
            id=args.id,
            choose_left_right=args.choose_left_right,
        )
        results["aro"].append({"score": aro_score})

    if args.eval_cola:
        print("Evaluating on COLA...")
        cola_score = evaluate_cola(
            model=flamingo,
            tokenizer=tokenizer,
            image_processor=image_processor,
            vis_embed_size=vis_embed_size,
            rank=args.rank,
            world_size=args.world_size,
            id=args.id,
        )
        results["cola"].append({"score": cola_score})


if __name__ == "__main__":
    main()

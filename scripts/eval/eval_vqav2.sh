LM_PATH=EleutherAI/pythia-1.4b
LM_TOKENIZER_PATH=EleutherAI/pythia-1.4b
VISION_ENCODER_NAME=ViT-L-14
VISION_ENCODER_PRETRAINED=datacomp_xl_s13b_b90k
VQAV2_IMG_PATH=/gpfs/u/home/LMCG/LMCGljnn/scratch/datasets/task/open_flamingo/okvqa/val2014
VQAV2_ANNO_PATH=/gpfs/u/home/LMCG/LMCGljnn/scratch/datasets/task/open_flamingo/vqav2/v2_mscoco_val2014_annotations.json
VQAV2_QUESTION_PATH=/gpfs/u/home/LMCG/LMCGljnn/scratch/datasets/task/open_flamingo/vqav2/v2_OpenEnded_mscoco_val2014_questions.json
CKPT_PATH=$1


RANDOM_ID=$$
torchrun --nnodes=1 --nproc_per_node=8 open_flamingo/eval/evaluate.py \
    --lm_path $LM_PATH \
    --lm_tokenizer_path $LM_TOKENIZER_PATH \
    --vision_encoder_path $VISION_ENCODER_NAME \
    --vision_encoder_pretrained $VISION_ENCODER_PRETRAINED \
    --checkpoint_path $CKPT_PATH \
    --vqav2_image_dir_path ${VQAV2_IMG_PATH} \
    --vqav2_questions_json_path ${VQAV2_QUESTION_PATH} \
    --vqav2_annotations_json_path ${VQAV2_ANNO_PATH} \
    --eval_vqav2 \
    --batch_size 2 \
    --id ${RANDOM_ID} \
    --dist

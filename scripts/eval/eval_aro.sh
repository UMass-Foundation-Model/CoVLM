LM_PATH=EleutherAI/pythia-1.4b
LM_TOKENIZER_PATH=EleutherAI/pythia-1.4b
VISION_ENCODER_NAME=ViT-L-14
VISION_ENCODER_PRETRAINED=datacomp_xl_s13b_b90k
CKPT_PATH=$1


RANDOM_ID=$$
torchrun --nnodes=1 --nproc_per_node=8 open_flamingo/eval/evaluate.py \
    --lm_path $LM_PATH \
    --lm_tokenizer_path $LM_TOKENIZER_PATH \
    --vision_encoder_path $VISION_ENCODER_NAME \
    --vision_encoder_pretrained $VISION_ENCODER_PRETRAINED \
    --checkpoint_path $CKPT_PATH \
    --eval_aro \
    --batch_size 1 \
    --id ${RANDOM_ID} \
    --dist

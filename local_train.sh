CAPTION_DATA=/gpfs/u/home/LMCG/LMCGljnn/scratch-shared/junyan/raw/blip2_all_data_ground/{000000..011698}.tar
PILE_DATA=/gpfs/u/home/LMCG/LMCGljnn/scratch-shared/junyan/raw/the_pile/{00000..01925}.tar
SAVE_DIR=checkpoints_local/103101
mkdir -p ${SAVE_DIR}
cp $0 ${SAVE_DIR}/

export TRANSFORMERS_OFFLINE=1
torchrun --nnodes=1 --nproc_per_node=6 --master_port=14288 open_flamingo/train/train.py \
--run_name ${SAVE_DIR} \
--vision_encoder_path ViT-L-14 \
--vision_encoder_pretrained datacomp_xl_s13b_b90k \
--lm_path EleutherAI/pythia-1.4b \
--tokenizer_path EleutherAI/pythia-1.4b \
--dataset_resampled \
--caption_shards ${CAPTION_DATA} \
--pile_shards ${PILE_DATA} \
--batch_size_caption 8 \
--batch_size_pile 8 \
--workers=4 \
--lr_scheduler constant \
--warmup_steps 1000 \
--num_steps 500000 \
--checkpoint_activations \
--delete_previous_checkpoint \
--gradient_accumulation_steps 1 \
--save_interval 100 \
--logging_steps 5 \
--skip_delete_pattern 1000 \
--precision amp_fp16 \
--learning_rate 1.0e-4 \
--max-length 1152 \
--loss_multiplier_det 0.025

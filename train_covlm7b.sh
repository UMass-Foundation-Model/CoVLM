LAION_DATA=llava_1_5/image/{00000..00611}.tar
PILE_DATA=llava_1_5/no_image/{00000..00040}.tar
SAVE_DIR=checkpoints/covlm7b
mkdir -p ${SAVE_DIR}
cp $0 ${SAVE_DIR}/

export TRANSFORMERS_OFFLINE=1
python open_flamingo/train/train.py \
--run_name ${SAVE_DIR} \
--vision_encoder_path ViT-L-14 \
--vision_encoder_pretrained datacomp_xl_s13b_b90k \
--lm_path liuhaotian/llava-v1.5-7b \
--tokenizer_path liuhaotian/llava-v1.5-7b \
--dataset_resampled \
--laion_shards ${LAION_DATA} \
--batch_size_laion 2 \
--workers=1 \
--lr_scheduler cosine \
--warmup_steps 500 \
--num_steps 10000 \
--checkpoint_activations \
--delete_previous_checkpoint \
--gradient_accumulation_steps 1 \
--save_interval 100 \
--logging_steps 1 \
--skip_delete_pattern 500 \
--precision amp_fp16 \
--learning_rate 2.0e-5 \
--add_visual_token \
--max-length 1024 \
--loss_multiplier_det 0.05 \
--add_box \
--expand \
--use_format_v2 \
--freeze_vision_encoder \
--load_detection_head_weight checkpoints/091701_pythiaS_previsual_fix/checkpoint_18000.pt \
--instruct \
--pile_freq 0

CONFIG="groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECK="/gpfs/u/home/LMCG/LMCGzhnf/scratch-shared/zfchen/code/GroundingDINO/checkpoints/groundingdino_swint_ogc.pth"
CUDA_VISIBLE_DEVICES=0 python demo/inference_on_laion.py \
  -c $CONFIG \
  -p $CHECK \
  -i .asset/cats.png \
  -o "outputs/0" \
  -t "a head of a cat." \
  --visualize

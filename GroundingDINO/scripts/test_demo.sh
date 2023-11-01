#CONFIG="groundingdino/config/GroundingDINO_SwinB.cfg.py"
#CHECK="/gpfs/u/home/LMCG/LMCGzhnf/scratch-shared/zfchen/code/GroundingDINO/checkpoints/groundingdino_swinb_cogcoor.pth"
CONFIG="groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECK="/gpfs/u/home/LMCG/LMCGzhnf/scratch-shared/zfchen/code/GroundingDINO/checkpoints/groundingdino_swint_ogc.pth"
CUDA_VISIBLE_DEVICES=6 python demo/inference_on_a_image.py \
  -c $CONFIG \
  -p $CHECK \
  -i .asset/cats.png \
  -o "outputs/0" \
  -t "a head of the cat" \

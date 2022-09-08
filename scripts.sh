### Train ###
CUDA_VISIBLE_DEVICES=1 python ZSSGAN/train.py \
    --size 512 \
    --batch 2 \
    --n_sample 4 \
    --output_dir training-runs/eg3d_Neanderthal \
    --lr 0.002 \
    --frozen_gen_ckpt D:/projects/IDE-3D/pretrained_models/ide3d/network-snapshot-014480.pkl \
    --iter 401 \
    --source_class Human \
    --target_class Neanderthal \
    --auto_layer_k 18 \
    --auto_layer_iters 1 \
    --auto_layer_batch 8 \
    --output_interval 100 \
    --clip_models ViT-B/32 ViT-B/16 \
    --clip_model_weights 1.0 1.0 \
    --save_interval 100 \
    --ide3d

### Render images ###
python gen_images.py \
        --network training-runs/eg3d_Neanderthal/checkpoint/000300.pkl \
        --seeds 58,96,174,180,179,185 \
        --trunc 0.7 \
        --outdir out

### Render videos ###
python gen_videos.py \
    --network training-runs/eg3d_Neanderthal/checkpoint/000300.pkl \
    --seeds 58,96,174,180,179,185 \
    --grid 3x2 \
    --trunc 0.7 \
    --outdir out \
    --image_mode image_seg 
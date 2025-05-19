# export TORCH_SDAA_LOG_LEVEL=trace

python ../train.py \
    --dataset_path /data/teco-data/imagenet \
    --batch_size 32 \
    --epochs 1 \
    --lr 0.01 \
    --amp \
    --save_path ../checkpoints \
    --max_step 100

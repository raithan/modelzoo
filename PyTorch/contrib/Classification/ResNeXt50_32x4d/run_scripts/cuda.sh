python ../train.py \
    --dataset_path /root/data/imagenet \
    --batch_size 16 \
    --epochs 1 \
    --lr 0.01 \
    --amp \
    --save_path ../checkpoints \
    --max_step 100 \
    --device cuda

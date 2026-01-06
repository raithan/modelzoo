#!/bin/bash
export TORCH_SDAA_AUTOLOAD=cuda_migrate
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_SDAA_CACHING_ALLOCATOR_TYPE="lifecycle"
TASK="MedSAM-ViT-B"

NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6789

python train_multi_cuda.py \
    --task_name "${TASK}" \
    --model_type vit_b \
    --tr_npy_path /data/application/gaolj/repo/MedSAM-main/data/npy/CT_Abd \
    --checkpoint /data/application/gaolj/repo/MedSAM-main/work_dir/MedSAM/sam_vit_b_01ec64.pth \
    --work_dir ./work_dir \
    --num_epochs 1 \
    --batch_size 1 \
    --grad_acc_steps 8 \
    --num_workers 4 \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${GPUS_PER_NODE}" \
    --node_rank ${NODE_RANK} \
    --init_method tcp://${MASTER_ADDR}:${MASTER_PORT}


wait ## Wait for the tasks on nodes to finish
echo "END TIME: $(date)"
#!/bin/bash
export TORCH_SDAA_AUTOLOAD=cuda_migrate
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_SDAA_CACHING_ALLOCATOR_TYPE="lifecycle"
export TORCH_SDAA_CONV2D_BACKWARD_USE_FP32=1
TASK="MedSAM-ViT-B"

NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6789

python train_multi_sdaa.py \
    --task_name "${TASK}" \
    --model_type vit_b \
    --tr_npy_path <your_path>/MedSAM/data/npy/CT_Abd \
    --checkpoint <your_path>/MedSAM/work_dir/MedSAM/sam_vit_b_01ec64.pth \
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
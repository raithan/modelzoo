#!/usr/bin/env bash
# 设置cuda_migrate环境变量
export TORCH_SDAA_AUTOLOAD=cuda_migrate
#export TORCH_SDAA_CACHING_ALLOCATOR_TYPE=lifecycle
#export PYTHONPATH=/data/suda-data/chenss/mmpretrain-main:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py --amp \
    $CONFIG \
    --launcher pytorch ${@:3}
    
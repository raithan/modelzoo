#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"
cd .. 
#安装依赖
pip install -r requirements.txt
echo "默认 1 个进程（1 GPU）；可以通过第一个参数覆盖，比如 ./test.sh 4"
NUM_PROC=${1:-1}
cd $script_path
echo "Using NUM_PROC=${NUM_PROC}"
echo "执行训练"
torchrun --nproc_per_node=${NUM_PROC} train.py \
    --data-dir /data/teco-data/imagenet \
    --model nest_small \
    --sched cosine \
    --epochs 2 \
    --warmup-epochs 5 \
    --lr 0.4 \
    --reprob 0.5 \
    --remode pixel \
    --batch-size 16 \
    --amp \
    -j 4 \
    2>&1 | tee sdaa.log
  # 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log

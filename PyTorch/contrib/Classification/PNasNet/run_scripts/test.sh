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
  --data-dir /data/teco-data/imagenet/ \
  --model pnasnet5large \
  -b 128 \
  --sched step \
  --epochs 450 \
  --decay-epochs 2.4 \
  --decay-rate .97 \
  --opt rmsproptf \
  --opt-eps .001 \
  -j 8 \
  --warmup-lr 1e-6 \
  --weight-decay 1e-5 \
  --drop 0.3 \
  --drop-path 0.2 \
  --model-ema \
  --model-ema-decay 0.9999 \
  --aa rand-m9-mstd0.5 \
  --remode pixel \
  --reprob 0.2 \
  --amp \
  --lr .016 \
  2>&1 | tee sdaa.log

  # 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
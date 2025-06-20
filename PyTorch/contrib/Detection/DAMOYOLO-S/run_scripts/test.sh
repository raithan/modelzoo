#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

# 安装依赖
echo "正在安装Python依赖..."
cd $script_path/../
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH

cd $script_path

#启动训练
python -m torch.distributed.launch --nproc_per_node=1 run_DAMOYOLO_S.py -f configs/damoyolo_tinynasL25_S.py 2>&1 | tee sdaa.log

#生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log

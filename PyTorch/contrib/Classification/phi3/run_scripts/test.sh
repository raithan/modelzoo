#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

#安装依赖
pip install -r requirements.txt

#执行训练
python train.py 

# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
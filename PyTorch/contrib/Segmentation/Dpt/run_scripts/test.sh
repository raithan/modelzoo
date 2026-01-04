#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

data_path="/data/suda-data/lf/ADEChallengeData2016"

#安装依赖
cd .. 
pip3 install  -U openmim 
pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
pip install opencv-python==4.7.0.72
pip install "mmcv==2.1.0"
mim install -e .
pip install -r requirements.txt
pip3 install numpy==1.24.3


cd $script_path

#执行训练
python run_dpt.py --config ../configs/dpt/dpt_vit-b16_8xb2-160k_ade20k-512x512.py \
    --launcher pytorch --nproc-per-node 4 --amp \
    --cfg-options "train_dataloader.dataset.data_root=$data_path" "val_dataloader.dataset.data_root=$data_path" 2>&1 | tee sdaa.log

# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log

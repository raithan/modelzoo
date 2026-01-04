#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

data_path="/data/teco-data/coco"
#该模型数据集路径需要修改configs/_base_/datasets/coco_detection.py中的data_root

#安装依赖
cd .. 
pip3 install  -U openmim 
pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
pip3 install opencv_python mmcv --no-deps
mim install -e .
pip install -r requirements.txt
pip3 install numpy==1.24.3

cd $script_path

#执行训练
python run_ssd300.py --config ../configs/ssd/ssd300_coco.py \
    --launcher pytorch --nproc-per-node 4 --amp 2>&1 | tee sdaa.log
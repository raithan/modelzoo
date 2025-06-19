#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo $script_path

#安装依赖
pip install -r ../requirements.txt
cd $script_path
# 数据集路径
data_path="/data/teco-data/COCO"


# 开启训练
python -m torch.distributed.launch --nproc_per_node=4 --use_env run_ConditionalDETR.py --coco_path /data/teco-data/COCO --output_dir output/conddetr_r50_epoch50  2>&1 | tee sdaa.log

#生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
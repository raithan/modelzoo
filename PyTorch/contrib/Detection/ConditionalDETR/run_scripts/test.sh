#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo $script_path

#安装依赖
pip3 install -r ../requirements.txt
cd $script_path
# 数据集路径
data_path="/data/teco-data/COCO"


# 如果在太初卡上开启分布式，设置nproc_per_node=4；日志文件会在主目录下生成类似train_log.json的日志；下面是开启训练的指令
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --coco_path /data/teco-data/COCO --output_dir output/conddetr_r50_epoch50

#生成loss对比图，原始生成的日志是json文件，请手动改一下；
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
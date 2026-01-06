#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径：$script_path"
data_path="/data/teco-data/coco2014/images"

# 安装依赖
echo" 此模型需要低版本numpy,检查numpy版本"
pip show numpy
pip install numpy==1.26.4

pip install -r requirements.txt
# 执行训练
python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/blip2/train/caption_coco_ft.yaml --options datasets.coco_caption.build_info.images.storage="$data_path" 2>&1 |tee sdaa.log

# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
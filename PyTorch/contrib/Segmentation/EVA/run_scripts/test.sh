#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

data_path="/data/teco-data/imagenet"

#安装依赖
cd .. 
pip install  -U openmim 
pip install git+https://gitee.com/xiwei777/mmengine_sdaa.git  
pip install opencv_python mmcv --no-deps
mim install -e . 
pip install -r requirements.txt
pip install git+https://github.com/Tecorigin/tcap_dllogger.git

cd $script_path

export TORCH_SDAA_AUTOLOAD=cuda_migrate

#执行训练
python run_eva.py --config ../configs/eva/eva-l-p14_8xb16_in1k-196px.py \
    --launcher pytorch --nproc-per-node 4 --amp \
    --cfg-options "train_dataloader.dataset.data_root=$data_path" "val_dataloader.dataset.data_root=$data_path" 2>&1 | tee sdaa.log


# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
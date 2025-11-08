#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

data_path="/data/teco-data/lvis"

#安装依赖
cd .. 
pip install  -U openmim 
pip install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
pip install opencv_python mmcv==2.1.0 --no-deps
mim install -e .
pip install -r requirements.txt
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install git+https://github.com/openai/CLIP.git

cd $script_path

export TORCH_SDAA_AUTOLOAD=cuda_migrate

#执行训练
python run_detic.py ../projects/Detic_new/detic_centernet2_r50_fpn_4x_lvis_boxsup.py \
    --nnodes 1     --nproc_per_node 1\
    --cfg-options "train_cfg.max_iters=200"  --cfg-options "train_cfg.val_interval=90000" 2>&1 | tee sdaa.log


# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
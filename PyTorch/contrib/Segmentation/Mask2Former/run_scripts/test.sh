#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

# 安装依赖
echo "正在安装Python依赖..."
cd $script_path/../
pip install  -U openmim 
pip install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
pip install opencv_python mmcv==2.1.0 --no-deps
mim install -e .
pip install -r requirements.txt
# pip install numpy==1.24.3

# 数据集路径设置
# 数据集路径,保持默认统一根目录即可
data_path="/data/teco-data/ade/ADEChallengeData2016"
# # 参数校验
# for para in $*
# do
#     if [[ $para == --data_path* ]];then
#         data_path=`echo ${para#*=}`
# done

# 训练参数配置
log_file="$script_path/sdaa.log"

# 启动训练
echo "开始训练..."
cd $script_path/

# export TORCH_SDAA_AUTOLOAD=cuda_migrate

python run_mask2former.py   ../configs/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py \
    --launcher pytorch \
    --cfg-options "train_dataloader.dataset.data_root=$data_path" "val_dataloader.dataset.data_root=$data_path" "train_cfg.max_iters=100" \
    2>&1 | tee sdaa.log


# 生成loss曲线图
echo "生成训练结果图表..."
python loss.py \
    --sdaa-log $log_file \
    --cuda-log ./cuda.log 

echo "训练完成！结果保存在:"
echo " - 训练日志: $log_file"
echo " - Loss曲线: $script_path/loss.jpg"
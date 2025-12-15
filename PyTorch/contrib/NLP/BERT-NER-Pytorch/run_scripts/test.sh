#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

# 安装依赖
echo "正在安装Python依赖..."
cd $script_path/../
pip install -r requirements.txt

# 数据集路径设置
# 数据集路径,保持默认统一根目录即可
data_path="/data/teco-data/BERT-NER_Pytorch"
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

export TORCH_SDAA_AUTOLOAD=cuda_migrate

python run_BERT-NER-Pytorch.py \
    --model_type=bert \
    --model_name_or_path=$data_path/prev_trained_model/bert-base-chinese \
    --task_name="cner" \
    --do_train \
    --do_lower_case \
    --data_dir=$data_path/datasets/cner/ \
    --train_max_seq_length=128 \
    --eval_max_seq_length=512 \
    --per_gpu_train_batch_size=24 \
    --per_gpu_eval_batch_size=24 \
    --learning_rate=3e-5 \
    --crf_learning_rate=1e-3 \
    --max_steps=100 \
    --logging_steps=1 \
    --save_steps=-1 \
    --output_dir=./outputs/cner_output/ \
    --overwrite_output_dir \
    --seed=42 \
    --local_rank=0 \
    2>&1 | tee $log_file

#此代码为100iters训练，如需长训请参考(./README.md)

# 生成loss曲线图
echo "生成训练结果图表..."
python loss.py \
    --sdaa-log $log_file \
    --cuda-log ./cuda.log 

echo "训练完成！结果保存在:"
echo " - 训练日志: $log_file"
echo " - Loss曲线: $script_path/loss.jpg"
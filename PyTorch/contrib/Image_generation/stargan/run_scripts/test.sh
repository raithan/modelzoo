#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

# 安装依赖
echo "正在安装Python依赖..."
cd $script_path/../
pip install -r requirements.txt

# 数据集路径设置
# 数据集路径,保持默认统一根目录即可
data_path="/data/teco-data/celeba-stargan"
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

python run_stargan.py \
    --mode train \
    --dataset CelebA \
    --num_iters 150 \
    --log_step 1 \
    --image_size 128 \
    --c_dim 5 \
    --sample_dir stargan_celeba/samples \
    --log_dir stargan_celeba/logs \
    --model_save_dir stargan_celeba/models \
    --result_dir stargan_celeba/results \
    --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
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
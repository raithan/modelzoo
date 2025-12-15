#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

# 安装依赖
echo "正在安装Python依赖..."
cd $script_path/../
apt-get update && apt-get install portaudio19-dev
pip install -r requirements.txt

# 数据集路径设置
# 数据集路径,保持默认统一根目录即可
data_path="/data/teco-data/MUSDB18-wav"
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

python run_mel_band_roformer.py \
    --model_type "mel_band_roformer" \
    --config_path "configs/config_musdb18_mel_band_roformer.yaml" \
    --results_path "results/" \
    --data_path "/data/teco-data/MUSDB18-wav/train" \
    --valid_path "/data/teco-data/MUSDB18-wav/test" \
    --num_workers 4 \
    --device_ids 0 \
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
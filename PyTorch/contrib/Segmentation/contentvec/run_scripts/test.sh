#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

# 安装依赖
echo "正在安装Python依赖..."
cd $script_path/../fairseq
python -m pip install --upgrade pip==24.0
python -m pip install ninja
python -m pip install --editable ./
python setup.py build_ext --inplace    
python -m pip install scipy
python -m pip install soundfile
python -m pip install praat-parselmouth
python -m pip install tensorboardX
python -m pip install numpy==1.26.4


# 数据集路径设置
# 数据集路径,保持默认统一根目录即可
data_path="/data/teco-data/LibriSpeech"
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

python run_san.py \
    --config_dir ./contentvec/config/contentvec \
    --config_name contentvec \
    --task.data /data/teco-data/LibriSpeech/metadata \
    --task.label_dir /data/teco-data/LibriSpeech/label \
    --task.spk2info /data/teco-data/LibriSpeech/metadata/spk2info.dict \
    --optimization.max_update 200 \
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
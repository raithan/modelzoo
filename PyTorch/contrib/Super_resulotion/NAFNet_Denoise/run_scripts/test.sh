#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

# 安装依赖
echo "正在安装Python依赖..."
cd $script_path/../
pip install -r requirements.txt
python setup.py develop --no_cuda_ext

# 数据集路径设置
# 数据集路径,保持默认统一根目录即可
data_path="/data/teco-data/SIDD"
# # 参数校验
# for para in $*
# do
#     if [[ $para == --data_path* ]];then
#         data_path=`echo ${para#*=}`
# done

# 训练参数配置
config_file="$script_path/../options/train/SIDD/NAFNet-width32.yml"
log_file="$script_path/sdaa.log"

# 启动训练
echo "开始训练..."
cd $script_path/
python run_NAFNet.py \
    --opt $config_file \
    --launcher pytorch \
    2>&1 | tee $log_file

#此代码为100iters训练，如需长训请修改$config_file路径下NAFNet-width32.yml

# 生成loss曲线图
echo "生成训练结果图表..."
python loss.py \
    --sdaa-log $log_file \
    --cuda-log ./cuda.log 

echo "训练完成！结果保存在:"
echo " - 训练日志: $log_file"
echo " - Loss曲线: $script_path/loss.jpg"
#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

# 安装依赖
echo "正在安装Python依赖..."
cd $script_path/../
pip install ultralytics
pip install -r requirements.txt

cd $script_path

# 数据集路径,保持默认统一根目录即可
#data_path="/data/imagnet"
# # 参数校验
# for para in $*
# do
#     if [[ $para == --data_path* ]];then
#         data_path=`echo ${para#*=}`
# done

#如长训请提供完整命令即可，如果需要100iter对齐，请在../ultralytics/engine/trainer.py中，将409行注释掉。

python ../train.py  # 验收测试用该脚本
# python run_demo.py

#生成loss对比图
python loss.py --sdaa-log new_sdaa.log --cuda-log cuda.log   #训练100步结束后../ultralytics/engine/trainer.py中会自动生成new_sdaa.log文件

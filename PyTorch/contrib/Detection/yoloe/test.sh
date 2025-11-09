#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

# 安装依赖
echo "正在安装Python依赖..."
echo "请检查numpy版本，如>1.26.4，请安装numpy==1.26.4"
pip install -r requirements.txt
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt
# 数据集路径设置
# 数据集路径,保持默认统一根目录即可
data_path="ultralytics/cfg/datasets/coco.yaml"
echo "若未下载数据集，请取消download部分的注释，同时修改数据集下载存放位置"
# 训练参数配置
log_file="$script_path/sdaa.log"
echo "如需修改训练相关参数，请参考/ultralytics/cfg/default.yaml（如amp : False）"
echo "如需设置多GPU训练，请在$script_path/train_pe.py中进行修改device的个数"
# 启动训练
echo "开始训练..."
cd $script_path/
python train_pe.py 2>&1 | tee $log_file

# 生成loss曲线图
echo "生成训练结果图表..."
python loss.py 

echo "训练完成！结果保存在:"
echo " - 训练日志: $log_file"
echo " - Loss曲线: $script_path/loss.jpg"
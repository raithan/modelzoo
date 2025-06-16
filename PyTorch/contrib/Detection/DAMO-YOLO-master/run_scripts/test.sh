#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

# 安装依赖
echo "正在安装Python依赖..."
cd $script_path/../
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
#step 2
# pip install cython
# pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI # for Linux

cd $script_path
# 数据集路径,保持默认统一根目录即可
# data_path="/data/teco-data/COCO"
# !!!!必须手动修改路径!!!!，请在DAMO-YOLO-master/damo/config/paths_catalog.py中修改链接（SymblockLink）


#训练测试:   原命令行：python -m torch.distributed.launch --nproc_per_node=1 tools/train.py -f configs/damoyolo_tinynasL25_S.py
#此代码为100iters训练，如需长训请将DAMO-YOLO-master/damo/apis/detector_trainer.py中的100步断点注释掉，然后在configs/damoyolo_tinynasL25_S.py中设置训练参数就好
python run_demo.py

#生成loss对比图
python loss.py --sdaa-log new_sdaa.log --cuda-log CUDA.log

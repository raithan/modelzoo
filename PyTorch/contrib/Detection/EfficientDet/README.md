# EfficientDet
## 1. 模型概述
EfficientDet 是一种高效且轻量级的目标检测模型，最早由 Google Brain 在 2020 年提出。它的设计理念是在保证高精度的同时大幅降低模型参数量和计算量，非常适合边缘设备和对实时性有要求的场景。

- 论文链接：https://arxiv.org/pdf/1911.09070
- 仓库链接：https://github.com/open-mmlab/mmdetection/tree/main/projects/EfficientDet

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
<MODLE EfficientDet>使用 COCO2017 数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/#download) 下载。

#### 2.2.2 处理数据集
具体配置方式可参考：https://blog.csdn.net/xzxg001/article/details/142465729。


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip3 install  -U openmim 
    pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
    pip3 install opencv_python mmcv --no-deps
    mim install -e .
    pip install -r requirements.txt
    ```
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
  ```
  cd <ModelZoo_path>/PyTorch/contrib/Detection/EfficientDet/run_scripts
  ```
2. 运行训练。该模型支持单机单卡。
  ```
  python run_EfficientDet.py --config ../projects/EfficientDet/configs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco.py --launcher pytorch --nproc-per-node 1 --amp --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/coco" "val_dataloader.dataset.data_root=/data/teco-data/coco"
  ```
    更多训练参数参考 run_scripts/argument.py
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: 0.2210767410210036
MeanAbsoluteError: -0.10302028477191925
Rule,mean_absolute_error -0.10302028477191925
pass mean_relative_error=0.2210767410210036 <= 0.05 or mean_absolute_error=-0.10302028477191925 <= 0.0002
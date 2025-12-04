# Cascade Mask R-CNN 
## 1. 模型概述
Cascade Mask R-CNN是一种多阶段级联目标检测框架，旨在逐步提高检测框的精度。
通过设计多个级联的检测器，每一级都基于上一阶段更准确的候选框进行训练和预测。
每一阶段使用不同的回归阈值，模型逐步细化定位结果。

- 论文链接：https://arxiv.org/abs/1906.09756
- 仓库链接：https://github.com/open-mmlab/mmdetection/tree/main/configs/cascade_rcnn

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
<MODLE Cascade Mask R-CNN>使用 COCO2017 数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/#download) 下载。

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
  cd <ModelZoo_path>/PyTorch/contrib/Detection/cascade_mask_rnn/run_scripts
  ```
2. 运行训练。该模型支持单机单卡。
  ```
  python run_cascade_mask_rnn.py --config ../configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py --launcher pytorch --nproc-per-node 1  --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/coco/" "val_dataloader.dataset.data_root=/data/teco-data/coco/"
  ```
  #sdaa开启amp会出现训练卡住，loss出现nan，训练速度很慢等现象，建议关闭amp，关闭amp后，可与cuda对齐
    更多训练参数参考 run_scripts/argument.py
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.015701795913289754
MeanAbsoluteError: -0.1368870270252228
Rule,mean_absolute_error -0.1368870270252228
pass mean_relative_error=-0.015701795913289754 <= 0.05 or mean_absolute_error=-0.1368870270252228 <= 0.0002
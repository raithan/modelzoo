# RetinaNet

## 1. 模型概述

- 论文链接：https://arxiv.org/abs/1708.02002
- 仓库链接：[[mmdetection/configs/retinanet at main · open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection/tree/main/configs/retinanet)](https://github.com/open-mmlab/mmdetection/tree/main/configs/yolo)

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

retinanet使用COCO数据集，该数据集为开源数据集。

#### 2.2.2 处理数据集


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
cd <ModelZoo_path>PyTorch/build-in/Detection/retinanet
  ```

2. 运行训练。该模型支持单机单卡。

  ```
python tools/train.py configs/retinanet/retinanet_r101-caffe_fpn_1x_coco.py
  ```



### 2.5 训练结果

输出训练loss曲线及结果

# SCNet
## 1. 模型概述
SCNet 是一种基于 Cascade R-CNN 的实例分割网络，通过在训练与推理阶段保持样本 IoU 分布一致性并引入特征传递与全局语义上下文模块，
协同优化分类、检测和分割子任务，从而在 COCO 上显著提升框和掩码的 AP 并加速推理速度

- 论文链接：https://arxiv.org/abs/2012.10150
- 仓库链接：https://github.com/open-mmlab/mmdetection/tree/main/configs/scnet

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
<MODLE SCNet>使用 COCO2017 数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/#download) 下载。

#### 2.2.2 处理数据集
1.具体配置方式可参考：https://github.com/facebookresearch/detectron2/blob/main/datasets/README.md
2.或者按下述：
datasets #根目录
  /coco
    /annotations
    /train2017
    /val2017
    /stuffthingmaps # 多出全景分割数据集，设置路径 SCNet/configs/htc/htc_r50_fpn_1x_coco.py
      /train2017
      /val2017


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
  cd <ModelZoo_path>/PyTorch/contrib/Detection/SCNet/run_scripts
  ```
2. 运行训练。该模型支持单机单卡。
  ```
  python run_SCNet.py --config ../configs/scnet/scnet_r50_fpn_1x_coco.py --launcher pytorch --nproc-per-node 1 --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/coco/" "val_dataloader.dataset.data_root=/data/teco-data/coco/"
  ```
  #sdaa开启amp训练极慢
    更多训练参数参考 run_scripts/argument.py
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.07914417403047153
MeanAbsoluteError: -1.000756893157959
Rule,mean_absolute_error -1.000756893157959
pass mean_relative_error=-0.07914417403047153 <= 0.05 or mean_absolute_error=-1.000756893157959 <= 0.0002
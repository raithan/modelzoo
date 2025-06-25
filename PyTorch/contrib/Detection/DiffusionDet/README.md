# DiffusioinDet
## 1. 模型概述
DiffusionDet 引入了**扩散模型（Diffusion Model）**思想，借助生成式模型强大的表达能力，采用反向扩散过程逐步“生成”检测框，提供了一种新的检测思路。

- 论文链接：https://arxiv.org/abs/2211.09788
- 仓库链接：https://github.com/open-mmlab/mmdetection/tree/main/projects/DiffusionDet

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
<MODLE DiffusionDet>使用 COCO2017 数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/#download) 下载。

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
  cd <ModelZoo_path>/PyTorch/contrib/Detection/DiffusioinDet/run_scripts
  ```
2. 运行训练。该模型支持单机单卡。
  ```
  python run_DiffusioinDet.py --config ../projects/DiffusioinDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py --launcher pytorch --nproc-per-node 1 --amp --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/coco" "val_dataloader.dataset.data_root=/data/teco-data/coco"
  ```
    更多训练参数参考 run_scripts/argument.py
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.012601468443809495
MeanAbsoluteError: -0.9119894218444824
Rule,mean_absolute_error -0.9119894218444824
pass mean_relative_error=-0.012601468443809495 <= 0.05 or mean_absolute_error=-0.9119894218444824 <= 0.0002
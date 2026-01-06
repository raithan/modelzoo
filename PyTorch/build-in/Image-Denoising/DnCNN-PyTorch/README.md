# DnCNN-PyTorch

## 1. 模型概述

这是TIP2017论文《超越高斯去噪：深度CNN图像去噪的残差学习》的PyTorch实现。DnCNN-PyTorch的代码主要从[GitHub]迁移和调整 [GitHub](https://github.com/SaoYan/DnCNN-PyTorch).

- 仓库链接：[pretrained-models.pytorch](https://github.com/SaoYan/DnCNN-PyTorch)
## 2. 快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 安装依赖：介绍模型依赖包。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 安装依赖

安装依赖:scikit-image  opencv-python h5py tensorboardX

### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。

执行以下命令，启动虚拟环境。

   ```
   conda activate torch_env
   ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。

   ```
   cd <ModelZoo_path>/PyTorch/build-in/Image-Denoising/DnCNN-PyTorch
   ```

2. 运行训练。该模型支持单卡单核组。

   训练DnCNN- s（已知噪声水平的DnCNN）

   ```
   python train.py --preprocess True --num_of_layers 17 --mode S --noiseL 25  --val_noiseL 25 --lr 0.005
   ```
   训练DnCNN- b（盲噪声水平DnCNN）
   
   ```
   python train.py --preprocess True --num_of_layers 20 --mode B --val_noiseL 25
   ```

   更多训练参数参考 README_EN.md

### 2.5 训练结果

输出训练loss曲线及结果（参考使用[loss.py](./loss.py)）: 
MeanRelativeError: 0.005401367366296666
MeanAbsoluteError: 0.0037739999999999705
Rule,mean_absolute_error 0.0037739999999999705
pass mean_relative_error=0.005401367366296666 <= 0.05 or mean_absolute_error=0.0037739999999999705 <= 0.0002

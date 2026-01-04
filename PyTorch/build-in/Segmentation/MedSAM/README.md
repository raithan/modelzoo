# MedSAM

## 1. 模型概述
这实现了在 FLARE22 数据集上对 MedSAM 的训练，主要基于 [MedSAM](https://github.com/bowang-lab/MedSAM/tree/main)仓库进行修改
- 仓库链接：[MedSAM](https://github.com/bowang-lab/MedSAM/tree/main)

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取权重&数据集：介绍如何获取训练所需的权重&数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备权重&数据集
#### 2.2.1 获取权重
SAM checkpoint：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

#### 2.2.2 处理权重
放置于work_dir/SAM/sam_vit_b_01ec64.pth

#### 2.2.3 获取数据集
FLARE2022：https://flare22.grand-challenge.org/ 

#### 2.2.4 处理数据集
python pre_CT_MR.py

split dataset: 80% for training and 20% for testing

分割数据集：80% 用于训练，20% 用于测试

adjust CT scans to soft tissue window level (40) and width (400)

调整 CT 扫描到软组织窗位（40）和窗宽（400）

max-min normalization  最大最小归一化

resample image size to 1024x1024

重采样图像大小到 1024x1024

save the pre-processed images and labels as npy files

将预处理后的图像和标签保存为 npy 文件

### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip install -r requirements.txt
    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/build-in/Segmentation/MedSAM
    ```

2. 运行训练。该模型支持单机单卡。

    ```
    修改train_multi_gpus.sh中的tr_npy_path和checkpoint路径
    sh train_multi_gpus_sdaa.sh

    ```

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./loss.py)）: 
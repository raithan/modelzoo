# SPACH

## 1. 模型概述
SPACH（Spatial Permutation and CHannel Mixing）是一种结合了卷积神经网络（CNN）、Transformer 和多层感知机（MLP）优势的混合架构，旨在设计高效且性能强大的视觉模型。
- SPACH ([A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP](https://arxiv.org/abs/2108.13002))
- 仓库链接：[SPACH](https://github.com/microsoft/SPACH)
- 其他配置参考readme_en.md

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
SPACH运行在ImageNet 1k上，这是一个来自ILSVRC挑战赛的广受欢迎的图像分类数据集。您可以点击[此链接](https://image-net.org/download-images)从公开网站中下载数据集。

#### 2.2.2 处理数据集
· 执行以下命令，解压训练数据集。
```
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

```
· 执行以下命令，解压测试数据并将图像移动到子文件夹中。
```
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

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
    cd <ModelZoo_path>/PyTorch/build-in/Classification/SPACH
    ```

2. 运行训练。该模型支持单机单卡。

    ```
    export TORCH_SDAA_AUTOLOAD=cuda_migrate  #自动迁移环境变量
    python -m torch.distributed.launch --nproc_per_node 1 --use_env main.py --epochs 1  --data-path <imagenet_path> --dist-eval --output_dir ./out
    ```

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./loss.py)）: 
MeanRelativeError: 1.2784229594112952e-07
MeanAbsoluteError: 8.842784965601424e-07
Rule,mean_relative_error 1.2784229594112952e-07
pass mean_relative_error=1.2784229594112952e-07 <= 0.05 or mean_absolute_error=8.842784965601424e-07 <= 0.0002



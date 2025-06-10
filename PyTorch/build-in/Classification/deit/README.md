# Data-Efficient architectures and training for Image classification

## 1. 模型概述

该仓库包含以下论文的 PyTorch 评估代码、训练代码和预训练模型：
DeiT Data-Efficient Image Transformers发表于 2021 年国际机器学习会议（ICML）[参考文献]。DeiT的代码主要从[GitHub]迁移和调整 [GitHub](https://github.com/facebookresearch/deit/tree/main).

- 仓库链接：[pretrained-models.pytorch](https://github.com/facebookresearch/deit/tree/main)

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

DeiT运行在ImageNet 1k上，这是一个来自ILSVRC挑战赛的广受欢迎的图像分类数据集。您可以点击[此链接](https://image-net.org/download-images)从公开网站中下载数据集。

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
3. 安装已经适配sdaa的timm==1.0.15。

    ```
    pip install timm-1.0.15-py3-none-any.whl
    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。

   ```
   cd <ModelZoo_path>/PyTorch/build-in/Classification/deit
   ```

2. 运行训练。

    该模型支持单核组

   ```
   python main.py --model deit_small_patch16_224 --batch-size 16 --data-path <imagenet_path>  --output_dir ./output --epochs 1
   ```

   更多训练参数参考 README_deit.md

### 2.5 训练结果

输出训练loss曲线及结果（参考使用[loss.py](./loss.py)）: 
MeanRelativeError: 0.0003944498653429388
MeanAbsoluteError: 0.002723999999999975
Rule,mean_relative_error 0.0003944498653429388
pass mean_relative_error=0.0003944498653429388 <= 0.05 or mean_absolute_error=0.002723999999999975 <= 0.0002

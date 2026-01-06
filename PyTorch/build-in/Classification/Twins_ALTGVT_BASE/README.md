# Twins
Twins 是美团和阿德莱德大学合作提出的视觉注意力模型，相关论文被 NeurIPS 2021 会议接收，代码已在 GitHub 上开源。该模型提出了 Twins-PCPVT 和 Twins-SVT 两种架构，在 ImageNet 分类、ADE20K 语义分割、COCO 目标检测等多个经典视觉任务中均取得了业界领先的结果。
## 1. 模型概述
- [![NeurIPS](https://img.shields.io/badge/NeurIPS2021-5kTlVBkzSRx-%238c1b13)](https://openreview.net/forum?id=5kTlVBkzSRx)
- 仓库链接：[Twins](https://github.com/Meituan-AutoML/Twins.git)
- 其他配置参考README_en.md

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
Twins运行在ImageNet 1k上，这是一个来自ILSVRC挑战赛的广受欢迎的图像分类数据集。您可以点击[此链接](https://image-net.org/download-images)从公开网站中下载数据集。

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
    #安装timm
    cd timm && pip3 install -e .
    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/build-in/Classification/Twins
    ```

2. 运行训练。该模型支持单机单卡。

    ```
    export TORCH_SDAA_AUTOLOAD=cuda_migrate  #自动迁移环境变量
    python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model alt_gvt_base --batch-size 32 --data-path <imagenet_path> --dist-eval --drop-path 0.3
    ```

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./loss.py)）: 
MeanRelativeError: 0.0016058156848888041
MeanAbsoluteError: 0.011143799782222104
Rule,mean_relative_error 0.0016058156848888041
pass mean_relative_error=0.0016058156848888041 <= 0.05 or mean_absolute_error=0.011143799782222104 <= 0.0002




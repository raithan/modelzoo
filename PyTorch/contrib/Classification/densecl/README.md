
# DenseCL
## 1. 模型概述
DenseCL（Dense Contrastive Learning）是由阿德莱德大学（The University of Adelaide）、同济大学（Tongji University）和字节跳动AI实验室（ByteDance AI Lab）联合提出的一种面向密集视觉预测任务的自监督预训练方法。其核心思想是在像素（或局部特征）级别进行对比学习，通过建模样本中局部特征之间的对应关系，实现更适用于下游密集任务（如目标检测、语义分割和实例分割）的视觉表征学习。与传统的图像级自监督学习方法（如MoCo-v2）相比，DenseCL在保持模型结构简洁、计算开销极低（仅比基线慢<1%）的前提下，在多个密集预测任务中展现出显著优越的迁移性能。例如，在PASCAL VOC目标检测任务中提升2.0% AP，在COCO目标检测中提升1.1% AP，在COCO实例分割中提升0.9% AP，在PASCAL VOC语义分割中提升3.0% mIoU，在Cityscapes语义分割中提升1.8% mIoU。这些结果表明，DenseCL能够有效弥补图像级与像素级任务之间的表征差异，成为密集视觉任务自监督预训练的一个重要进展和代表性方法之一。



- 论文链接：[[2011.09157v2\]]Dense contrastive learning for self-supervised visual pre-training(https://arxiv.org/abs/2011.09157)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/densecl

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
densecl 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
  cd <ModelZoo_path>/PyTorch/contrib/Classification/densecl/run_scripts
  ```

2. 运行训练。该模型支持单机单卡。
  ```
  python run_densecl.py --config ../configs/densecl/benchmarks/resnet50_8xb32-linear-steplr-100e_in1k.py --launcher pytorch --nproc-per-node 4 --amp --cfg-options "train_dataloader.dataset.data_root=<imagenet_path>" "val_dataloader.dataset.data_root=<imagenet_path>"
  ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果

|模型             |    数据集      |    sdaa结果       |   cuda结果         |  sdaa耗时    |    
|:---------------|:--------------:|:-----------------:|:-----------------:|:-------------:|
|densecl    |imagenet-1k  |63.5140     |  63.50            |3d         |
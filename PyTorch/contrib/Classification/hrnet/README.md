# HRNet
## 1. 模型概述
HRNet（High-Resolution Network）是由微软亚洲研究院（Microsoft Research Asia）联合多所高校提出的一种用于视觉识别任务的深度高分辨率表征学习框架。其核心思想在于在整个网络处理过程中始终保持高分辨率特征表示，并通过并行连接高低分辨率卷积流以及跨分辨率信息交互机制，实现更丰富语义信息与更精确空间定位的特征融合。与传统方法（如ResNet、VGGNet）先提取低分辨率特征再上采样的方式不同，HRNet从网络初始阶段即同时维护多个分辨率分支，并在网络的每一层中进行跨分辨率的信息融合，从而在整个流程中生成语义更丰富、空间更精确的特征表示。该方法在包括人体姿态估计、语义分割、目标检测等多个视觉任务中均展现出显著的性能优势，成为高分辨率视觉识别任务的代表性骨干网络之一。同时，HRNet具有良好的通用性和扩展性，已被广泛应用于学术研究与工业落地中



- 论文链接：[[1908.07919v2\]]Deep High-Resolution Representation Learning for Visual Recognition(https://arxiv.org/abs/1908.07919v2)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/hrnet

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
hrnet使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
  cd <ModelZoo_path>/PyTorch/contrib/Classification/hrnet/run_scripts
  ```

2. 运行训练。该模型支持单机单卡。
  ```
  python hrnet.py --config ../configs/hrnet/hrnet-w18_4xb32_in1k.py --launcher pytorch --nproc-per-node 4 --amp --cfg-options "train_dataloader.dataset.data_root=<imagenet_path>" "val_dataloader.dataset.data_root=<imagenet_path>"
  ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果

|模型             |    数据集      |    sdaa结果       |   cuda结果         |  sdaa耗时    |    
|:---------------|:--------------:|:-----------------:|:-----------------:|:-------------:|
|hrnet    |imagenet-1k  |76.152     |  76.75            |4d         |
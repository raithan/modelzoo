
# MobileNetV3
## 1. 模型概述
MobileNetV3 是由谷歌大脑（Google Brain）团队提出的一种高效轻量级卷积神经网络（CNN），其核心思想是通过引入深度可分离卷积（Depthwise Separable Convolutions）、神经架构搜索（Neural Architecture Search, NAS）以及创新的轻量级注意力模块（如Squeeze-and-Excite的改进）和激活函数（如h-swish），在保持较高精度的前提下，显著降低模型的计算复杂度和参数量，优化移动端和嵌入式设备的推理速度与能效。MobileNetV3在2019年发布，通过平衡模型大小、计算成本和分类/检测性能，成为移动端深度学习部署的重要里程碑和代表性模型之一。



- 论文链接：[[1905.02244v5\]]Searching for MobileNetV3(https://arxiv.org/abs/1905.02244)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/mobilenet_v3

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
mobilenet_v3 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
  cd <ModelZoo_path>/PyTorch/contrib/Classification/mobilenet_v3/run_scripts
  ```

2. 运行训练。该模型支持单机单卡。
  ```
  python run_mobilenet_v3.py --config ../configs/mobilenet_v3/mobilenet-v3-small_8xb128_in1k.py --launcher pytorch --nproc-per-node 4 --amp --cfg-options "train_dataloader.dataset.data_root=<imagenet_path>" "val_dataloader.dataset.data_root=<imagenet_path>"
  ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果

|模型             |    数据集      |    sdaa结果       |   cuda结果         |  sdaa耗时    |    
|:---------------|:--------------:|:-----------------:|:-----------------:|:-------------:|
|mobilenet_v3    |imagenet-1k  |64.1060028     |  66.68            |7d4h         |
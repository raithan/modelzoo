
# ConvMixer
## 1. 模型概述
尽管卷积网络多年来一直是视觉任务的主导架构，但最近的实验表明，基于 Transformer 的模型，尤其是 Vision Transformer (ViT)，在某些场景下可能超越其性能。然而，由于 Transformer 中自注意力层的平方复杂度，ViT 需要利用图像块嵌入（将图像的小区域组合成单个输入特征）才能应用于更大的图像尺寸。这就引出了一个问题：ViT 的性能是源于其本质上更强大的 Transformer 架构，还是至少部分归功于使用了图像块作为输入表示？在本文中，我们提出了一些支持后者的证据：具体来说，我们提出了 ConvMixer，这是一个极其简单的模型。其设计理念与 ViT 以及更基础的 MLP-Mixer 相似，即：直接处理图像块作为输入；分离空间维度混合与通道维度混合；在整个网络中保持相同的特征图尺寸和分辨率。然而，与它们不同的是，ConvMixer 仅使用标准的卷积操作来实现这些混合步骤。尽管其结构简单，但我们证明，在参数量相近且数据集大小相当的情况下，ConvMixer 不仅超越了经典的视觉模型（如 ResNet），其性能也优于 ViT、MLP-Mixer 及其部分变体。

- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/convmixer

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
ConvMixer 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
  cd <ModelZoo_path>/PyTorch/contrib/Classification/ConvMixer/run_scripts
  ```

2. 运行训练。该模型支持单机单卡。
  ```
  python run_convmixer.py --config ../configs/convmixer/convmixer-768-32_10xb64_in1k.py --launcher pytorch --nproc-per-node 1 --amp --cfg-options "train_dataloader.dataset.data_root=<imagenet_path>" "val_dataloader.dataset.data_root=<imagenet_path>" "train_dataloader.batch_size=32"  "val_dataloader.batch_size=32" 
  ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）
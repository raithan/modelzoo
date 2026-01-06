
# **MobileNetV4**
## 1. 模型概述  
MobileNetV4 (MNv4) 是 Danfeng Qin 等人于 2024 年提出的新一代移动视觉网络，旨在为 异构移动生态系统（包括 CPU、GPU、DSP 以及专用神经加速器如 Apple Neural Engine 和 Google EdgeTPU 等）提供通用、极致高效且高精度的 backbone 架构。与早期 MobileNet 系列相比，MobileNetV4 在架构块设计、硬件感知搜索、注意力机制优化与知识蒸馏等方面做出了多项创新，从而在多个设备上实现了Pareto 最优的性能／延迟权衡。最终的高端变体在 ImageNet-1K 分类上达到了约 87% top-1 精度，在 Pixel 8 EdgeTPU 上推理仅需约 3.8 ms，展现了出色的实用性能。
> **论文链接**：https://arxiv.org/abs/2404.10518
> **仓库链接**：https://github.com/huggingface/pytorch-image-models  

## 2. 快速开始  
使用本模型执行训练的主要流程如下：  
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。  
2. 获取数据集：介绍如何获取训练所需的数据集。  
3. 构建环境：介绍如何构建模型运行所需要的环境。  
4. 启动训练：介绍如何运行训练。  

### 2.1 基础环境安装  

请参考基础环境安装章节，完成训练前的基础环境检查和安装。  

### 2.2 准备数据集  

### Step 1: Dataset Preparation

#### 2.2.1 获取数据集
ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

#### 2.2.2 处理数据集
具体配置方式可参考：https://blog.csdn.net/xzxg001/article/details/142465729。

### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境  
1. 执行以下命令，启动虚拟环境。  
    ```
    conda activate torch_env  
    ```
2. 安装python依赖  
    ```
    cd <ModelZoo_path>/PyTorch/contrib/pytorch_image_model/mobilenetv4/
    pip install -r requirements.txt
    pip install -e .
    ```
### 2.4 启动训练  
1. 在构建好的环境中，进入训练脚本所在目录。  
    ```
    cd <ModelZoo_path>/PyTorch/contrib/pytorch_image_model/mobilenetv4/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。

    -  单机单卡
    ```
    python run_mobilenetv4.py \
    --data-dir /data/teco-data/imagenet \
    --model mobilenetv4_conv_small \
    --sched cosine \
    --epochs 1 \
    --warmup-epochs 5 \
    --lr 0.4 \
    --reprob 0.5 \
    --remode pixel \
    --batch-size 16 \
    --amp \
    -j 4 \
    --log-interval 1 \
    2>&1 | tee sdaa.log
   ```
    更多训练参数参考[README](run_scripts/README.md)

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 
![训练loss曲线](./run_scripts/loss.jpg)

MeanRelativeError: 0.01625552727503719
MeanAbsoluteError: 0.11228748831418481
Rule,mean_relative_error 0.01625552727503719
pass mean_relative_error=0.01625552727503719 <= 0.05 or mean_absolute_error=0.11228748831418481 <= 0.0002

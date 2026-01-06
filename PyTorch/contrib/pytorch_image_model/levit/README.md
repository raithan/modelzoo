
# **LeViT**
## 1. 模型概述  
LeViT (a Vision Transformer in ConvNet’s Clothing for Faster Inference) 是 Ben Graham 等人于 2021 年提出的一种 混合视觉变换器（hybrid Vision Transformer）架构，旨在在 图像分类任务中实现优异的速度 / 精度折衷（speed/accuracy trade-off）。与标准 Vision Transformer（ViT）直接在高分辨率输入上进行全局自注意力计算不同，LeViT 借鉴了卷积网络（CNN）的 分层特征映射（activation maps with decreasing resolutions） 和高效结构，同时融合注意力机制，使其在真实设备（如 CPU / GPU）上推理速度显著提升，而保持与当时主流模型相当的分类精度。该设计尤其适合对 推理延迟和资源效率敏感的实际应用场景。
> **论文链接**：https://arxiv.org/abs/2104.01136
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
    cd <ModelZoo_path>/PyTorch/contrib/pytorch_image_model/levit/
    pip install -r requirements.txt
    pip install -e .
    ```
### 2.4 启动训练  
1. 在构建好的环境中，进入训练脚本所在目录。  
    ```
    cd <ModelZoo_path>/PyTorch/contrib/pytorch_image_model/levit/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。

    -  单机单卡
    ```
    python run_levit.py \
    --data-dir /data/teco-data/imagenet \
    --model levit_128 \
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

MeanRelativeError: -0.0007625029864648154
MeanAbsoluteError: -0.005715148283703493
Rule,mean_absolute_error -0.005715148283703493
pass mean_relative_error=-0.0007625029864648154 <= 0.05 or mean_absolute_error=-0.005715148283703493 <= 0.0002

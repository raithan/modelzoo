# EfficientNet V2
## 1. 模型概述
EfficientNet V2，这是一类新的卷积网络，训练速度更快，参数效率也优于以往模型。为了开发这一系列模型，我们结合了训练感知的神经结构搜索和缩放，共同优化训练速度和参数效率。这些模型在搜索空间中进行了搜索，并加入了如Fused-MBConv等新作。我们的实验表明，EfficientNetV2模型的训练速度远快于最先进模型，且其规模可达6.8倍。

- 论文链接：[[2104.00298\]]EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks(https://arxiv.org/abs/2104.00298)
- 仓库链接：https://github.com/huggingface/pytorch-image-models?tab=readme-ov-file#train-validation-inference-scripts

使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
EfficientNet V2使用 ImageNet数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/efficientnet_V2/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   torchrun --nproc_per_node=${NUM_PROC} train.py \
    --data-dir /data/teco-data/imagenet \
    --model efficientnetv2_s \
    --sched cosine \
    --epochs 2 \
    --warmup-epochs 5 \
    --lr 0.4 \
    --reprob 0.5 \
    --remode pixel \
    --batch-size 16 \
    --amp \
    -j 4 \
    2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/arguments.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: 0.0010747547297777053
MeanAbsoluteError: 0.007163562396965404
Rule,mean_relative_error 0.0010747547297777053
pass mean_relative_error=0.0010747547297777053 <= 0.05 or mean_absolute_error=0.007163562396965404 <= 0.0002

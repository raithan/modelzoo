# NesT
## 1. 模型概述
NesT，层级结构在近期的视觉变换器中很流行，但它们需要复杂的设计和庞大的数据集才能有效运行。NesT将基本局部变换器嵌套在不重叠的图像块上，并以分层方式聚合这些内容的理念。我们发现，块聚合功能在实现跨块非本地信息通信中起着关键作用。这一观察促使我们设计了一个简化架构，只需对原始视觉变换器进行小幅代码修改。所提的审慎选择设计有三方面的好处：（1）NesT收敛更快，且训练数据需求大幅减少，即可在ImageNet和CIFAR等小型数据集上实现良好泛化;（2）在将关键思想扩展到图像生成时，NesT会导致一个强解码器，即8×比之前基于变压器的发电机更快;

- 论文链接：[[2105.12723\]]Nested Hierarchical Transformer: Towards Accurate, Data-Efficient and Interpretable Visual Understanding(https://arxiv.org/abs/2105.12723)
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
NesT使用 ImageNet数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/NesT/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   torchrun --nproc_per_node=${NUM_PROC} train.py \
    --data-dir /data/teco-data/imagenet \
    --model nest_small \
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

MeanRelativeError: -0.0038046755978157778
MeanAbsoluteError: -0.027139706186728903
Rule,mean_absolute_error -0.027139706186728903
pass mean_relative_error=-0.0038046755978157778 <= 0.05 or mean_absolute_error=-0.027139706186728903 <= 0.0002
# RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space

## 1. 模型概述
这是用于知识图谱嵌入（KGE）的RotatE模型的 PyTorch 实现。我们提供了一个工具包，该工具包实现了多种流行 KGE 模型的前沿性能。该工具包效率极高，能够在单个 GPU 上于数小时内完成大型 KGE 模型的训练。
- 仓库链接：[KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
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
./data/下
#### 2.2.2 处理数据集
无需处理

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
    cd <ModelZoo_path>/PyTorch/build-in/others/RotatE
    ```

2. 运行训练。该模型支持单机单核组。

    ```
    export TORCH_SDAA_AUTOLOAD=cuda_migrate  #自动迁移环境变量
    python -u codes/run.py --do_train \
        --cuda \
        --do_valid \
        --do_test \
        --data_path data/FB15k \
        --model RotatE \
        -n 256 -b 1024 -d 1000 \
        -g 24.0 -a 1.0 -adv \
        -lr 0.0001 --max_steps 150000 \
        -save models/RotatE_FB15k_0 --test_batch_size 16 -de
    ```

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./loss.py)）




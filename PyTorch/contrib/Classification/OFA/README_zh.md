# OFA
## 1. 模型概述
OFA（One-For-All）是一个统一的多模态预训练模型，通过单一的模型架构解决多种多模态任务。它采用了任务无关的设计理念，将不同模态的任务统一表示为序列到序列的生成问题。

- 论文链接：[[2202.03052\]]OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework(https://arxiv.org/abs/2202.03052)
- 仓库链接：https://github.com/OFA-Sys/OFA?tab=readme-ov-file
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
OFA使用Refcoco数据集，该数据集为开源数据集，可从 [Refcoco](https://ofa-beijing.oss-cn-beijing.aliyuncs.com/datasets/refcoco_data/refcoco_data.zip) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/OFA/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   bash ./train_refcoco_prefix.sh
   ```
    更多训练参数参考 run_scripts/README.md

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: 0.9993047678654691
MeanAbsoluteError: 0.06904587499999999
Rule,mean_absolute_error 0.06904587499999999
fail mean_relative_error=0.9993047678654691 <= 0.05 or mean_absolute_error=0.06904587499999999 <= 0.0002
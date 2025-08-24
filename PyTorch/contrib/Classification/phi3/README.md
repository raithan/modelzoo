# Phi-3
## 1. 模型概述
Phi-3 是由微软公司开发的一个开创性的小型语言模型（Small Language Model, SLM） 系列。它的核心理念是挑战“越大越好”的传统观念，旨在通过极高质量的训练数据、精心的模型架构设计和严格的训练流程，在参数规模显著小于大型模型（如 GPT-4、Claude 3）的情况下，实现与之媲美的卓越性能。

- 仓库链接：https://github.com/datawhalechina/self-llm/blob/master
## 2. 快速开始
使用本模型执行Lora微调训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 微调模型文件：介绍训练所需要的微调训练文件下载。
3. 获取数据集：介绍如何获取训练所需的数据集。
4. 构建环境：介绍如何构建模型运行所需要的环境。
5. 启动训练：介绍如何运行训练。
6. 量化模型。

### 2.1 基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 微调基础模型文件下载
Phi-3 微调基础大模型选择Phi-3-mini-4k-instruct，模型文件可以在https://huggingface.co/，或者huggingface镜像网站HF-Mirror - Huggingface 镜像站，或者魔搭社区进行下载，此处用魔搭社区的git进行下载。

git clone https://www.modelscope.cn/LLM-Research/Phi-3-mini-4k-instruct.git

### 2.3 准备数据集
#### 2.2.1 获取数据集
Lora微调，此处使用中文数据集huanhuan数据集，该数据集为开源数据集，可从 [huanhuan](https://github.com/datawhalechina/self-llm/tree/master/dataset) 下载。

### 2.4 构建环境

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/phi3/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   python train.py 
   ```

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./loss.py)）: 

![loss_compare](./loss.jpg)

MeanRelativeError: -0.03429066357364151
MeanAbsoluteError: -0.10998400000000004
Rule,mean_absolute_error -0.10998400000000004
pass mean_relative_error=-0.03429066357364151 <= 0.05 or mean_absolute_error=-0.10998400000000004 <= 0.0002

## 3.量化模型
微调好的模型文件合并，具体推理效果和训练时间和数据集有关，详情操作请参考：https://github.com/datawhalechina/self-llm/blob/master/models/phi-3/04-Phi-3-mini-4k-Instruct%20Lora%20%E5%BE%AE%E8%B0%83.md
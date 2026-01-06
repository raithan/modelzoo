# ollama
## 1. 模型概述
ollama 是一个开源的大型语言模型服务工具，旨在帮助用户快速在本地运行大模型。通过简单的安装指令，用户可以通过一条命令轻松启动和运行开源的大型语言模型。它提供了一个简洁易用的命令行界面和服务器，专为构建大型语言模型应用而设计。用户可以轻松下载、运行和管理各种开源 LLM。与传统 LLM 需要复杂配置和强大硬件不同，Ollama 能够让用户在消费级的PC上体验LLM的强大功能。

- 仓库链接：https://gitcode.com/datawhalechina/handy-ollama?utm_source=highlight_word_gitcode&word=ollama
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
ollama微调基础大模型选择Chinese-Mistral-7B-Instruct-v0.1，模型文件可以在https://huggingface.co/，或者huggingface镜像网站HF-Mirror - Huggingface 镜像站，或者魔搭社区进行下载，此处用魔搭社区的git进行下载。
git clone https://www.modelscope.cn/itpossible/Chinese-Mistral-7B-Instruct-v0.1.git

### 2.3 准备数据集
#### 2.2.1 获取数据集
Lora微调，此处使用中文数据集ruozhiba数据集，该数据集为开源数据集，可从 [ruozhiba](https://huggingface.co/datasets/kigner/ruozhiba-llama3) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/ollama/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   python train.py 2>&1 | tee sdaa.log
   ```
    更多训练参数参考 ./argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./loss.py)）: 

![loss_compare](./loss.jpg)

MeanRelativeError: nan
MeanAbsoluteError: -0.009643478260869573
Rule,mean_absolute_error -0.009643478260869573
pass mean_relative_error=nan <= 0.05 or mean_absolute_error=-0.009643478260869573 <= 0.0002

## 3.量化模型
微调好的模型文件合并，具体推理效果和训练时间和数据集有关，详情操作请参考：https://blog.csdn.net/spiderwower/article/details/138755776
# DeiT-III
## 1. 模型概述
BERTSUM 是一种基于 Transformer 架构的文本摘要模型，通过结合预训练语言模型 BERT（Bidirectional Encoder Representations from Transformers）与序列建模技术，实现了高效的文本摘要生成。该模型在抽取式（Extractive）和生成式（Abstractive）两种摘要任务中均表现出色，尤其在长文本处理、信息检索和学术研究等领域具有广泛应用价值。

- 论文链接：[Fine-tune BERT for Extractive Summarization](https://arxiv.org/pdf/1903.10318.pdf)
- 仓库链接：https://github.com/nlpyang/BertSum
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
链接 https://drive.google.com/open?id=1x0d61LP9UAN389YN00z0Pv-7jQgirVg6


解压bertsum_data.zip到bert_data


#### 2.2.2 处理数据集
具体配置方式可参考：https://github.com/nlpyang/BertSum?tab=readme-ov-file#option-2-process-the-data-yourself


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
3. 添加环境变量。

```
export TORCH_SDAA_AUTOLOAD=cuda_migrate
export HF_ENDPOINT=https://hf-mirror.com
```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/NLP/BertSum/src
    ```

2. 运行训练。该模型支持单机单卡训练。

   ```
    sh sdaa_1card_test.sh
   ```
    注意：第一次启动训练会下载数据集，建议将visible_gpus和gpu_ranks改为0，下载成功后中断程序，再将visible_gpus和gpu_ranks恢复原状。
    
    更多训练参数参考 src/train.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./src/xnet.py)）: 

自测log位于logs文件夹下

sdaa_xent 数据的均值为: 3.9912

cuda_xent 数据的均值为: 3.9938
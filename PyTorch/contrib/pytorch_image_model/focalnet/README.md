
# **FocalNet**
## 1. 模型概述  
Focal Modulation Networks（简称 FocalNet）由 Jianwei Yang、Chunyuan Li、Xiyang Dai、Lu Yuan、Jianfeng Gao 等人在 2022 年提出，是一种“用焦点调制 (focal modulation) 完全替代自注意 (self-attention, SA)” 的视觉网络 (vision backbone)。它针对视觉任务 (分类 / 检测 /分割) 中 token（patch / feature map）之间的交互 (interaction) 提出了一种新的机制 — 不再依赖于 Transformer 中典型的 query-key-value 自注意力计算，而是通过层次化上下文聚合 + 动态调制 (modulation) 来实现高效且表达力强的特征交互。
> **论文链接**：https://arxiv.org/abs/2203.11926
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
    cd <ModelZoo_path>/PyTorch/contrib/pytorch_image_model/focalnet/
    pip install -r requirements.txt
    pip install -e .
    ```
### 2.4 启动训练  
1. 在构建好的环境中，进入训练脚本所在目录。  
    ```
    cd <ModelZoo_path>/PyTorch/contrib/pytorch_image_model/focalnet/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。

    -  单机单卡
    ```
    python run_focalnet.py \
    --data-dir /data/teco-data/imagenet \
    --model focalnet_base_lrf \
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

MeanRelativeError: 0.0019037652340935754
MeanAbsoluteError: 0.012350294849660137
Rule,mean_relative_error 0.0019037652340935754
pass mean_relative_error=0.0019037652340935754 <= 0.05 or mean_absolute_error=0.012350294849660137 <= 0.0002

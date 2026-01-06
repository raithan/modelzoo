
# **BEiT**
## 1. 模型概述  
**BEiT（BERT Pre-Training of Image Transformers）**由 Hangbo Bao、Li Dong、Furu Wei 等人提出，旨在为视觉 Transformer 设计一种类似 NLP 中 BERT 的统一自监督预训练方法。BEiT 将图像划分为固定大小的 patch，并在预训练阶段通过“图像分块掩码（masked image modeling）+ 离散视觉 token 预测”的方式学习图像表示。其核心创新在于引入视觉 tokenizer（如 VQ-VAE）将图像编码为离散视觉 token，从而使 ViT 能够以“掩码预测”方式进行预训练，显著提升了下游分类、分割等任务的特征表达能力与迁移性能。
> **论文链接**：https://arxiv.org/abs/2106.08254
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
    cd <ModelZoo_path>/PyTorch/contrib/pytorch_image_model/beit/
    pip install -r requirements.txt
    pip install -e .
    ```
### 2.4 启动训练  
1. 在构建好的环境中，进入训练脚本所在目录。  
    ```
    cd <ModelZoo_path>/PyTorch/contrib/pytorch_image_model/beit/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。

    -  单机单卡
    ```
    python run_beit.py \
    --data-dir /data/teco-data/imagenet \
    --model beit_base_patch16_224 \
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

MeanRelativeError: 1.4155326668947839e-05
MeanAbsoluteError: 9.774689627165842e-05
Rule,mean_relative_error 1.4155326668947839e-05
pass mean_relative_error=1.4155326668947839e-05 <= 0.05 or mean_absolute_error=9.774689627165842e-05 <= 0.0002

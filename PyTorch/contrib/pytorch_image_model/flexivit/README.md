
# **FlexiViT**
## 1. 模型概述  
FlexiViT 是由 Google Research 的 Lucas Beyer 等人于 2022 年提出的一种改进版视觉 Transformer (ViT) 架构，其目标是解决传统 ViT 对 patch size（图像切分块大小）高度敏感的问题。传统 ViT 在训练时使用固定的 patch 大小 (例如 16×16)，而如果改变 patch 大小 (patch size) 通常需要重新训练模型。FlexiViT 通过在训练阶段随机化 patch size，并共享同一组权重，得到一个“一模型多 patch-size (multi patch-size)”的 Transformer。这样在部署阶段，可以根据计算资源 /速度 /准确度需求动态调整 patch size，而不需要重新训练模型。
> **论文链接**：https://arxiv.org/abs/2212.08013
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
    cd <ModelZoo_path>/PyTorch/contrib/pytorch_image_model/flexivit/
    pip install -r requirements.txt
    pip install -e .
    ```
### 2.4 启动训练  
1. 在构建好的环境中，进入训练脚本所在目录。  
    ```
    cd <ModelZoo_path>/PyTorch/contrib/pytorch_image_model/flexivit/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。

    -  单机单卡
    ```
    python run_flexivit.py \
    --data-dir /data/teco-data/imagenet \
    --model flexivit_base \
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

MeanRelativeError: -0.0024478048044573536
MeanAbsoluteError: -0.019250614808337525
Rule,mean_absolute_error -0.019250614808337525
pass mean_relative_error=-0.0024478048044573536 <= 0.05 or mean_absolute_error=-0.019250614808337525 <= 0.0002

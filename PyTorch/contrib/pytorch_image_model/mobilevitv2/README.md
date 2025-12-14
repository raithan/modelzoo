
# **MobileViT**
## 1. 模型概述  
Separable Self‑attention for Mobile Vision Transformers 由 Sachin Mehta 与 Mohammad Rastegari 于 2022 年提出，旨在提升现有轻量级视觉 Transformer（特别是 MobileViT）在移动端设备的推理效率与速度。MobileViT 等移动视觉 Transformer 在资源受限设备（如手机、嵌入式设备）上的性能受限于自注意力机制（Multi‑Head Self‑Attention, MHA）的高复杂度，而本工作提出了一种**可分离自注意力（Separable Self‑Attention）**机制，以降低计算开销并保持良好的准确性，从而更适合移动视觉场景（classification/detection 等）。
> **论文链接**：https://arxiv.org/abs/2206.02680
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
    cd <ModelZoo_path>/PyTorch/contrib/pytorch_image_model/mobilevitv2/
    pip install -r requirements.txt
    pip install -e .
    ```
### 2.4 启动训练  
1. 在构建好的环境中，进入训练脚本所在目录。  
    ```
    cd <ModelZoo_path>/PyTorch/contrib/pytorch_image_model/mobilevitv2/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。

    -  单机单卡
    ```
    python run_mobilevitv2.py \
    --data-dir /data/teco-data/imagenet \
    --model mobilevitv2_050 \
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

MeanRelativeError: 0.00010264481990377813
MeanAbsoluteError: 0.0006963380492559754
Rule,mean_relative_error 0.00010264481990377813
pass mean_relative_error=0.00010264481990377813 <= 0.05 or mean_absolute_error=0.0006963380492559754 <= 0.0002

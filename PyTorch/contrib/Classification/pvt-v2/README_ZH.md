# PVT-V2 (Improved Pyramid Vision Transformer)
## 1. 模型概述
PVT-V2 (Improved Pyramid Vision Transformer)，本研究通过改进原始的金字塔视觉变换器（PVT v1）提出了新的基线，新增了三种设计，包括（1）线性复杂度注意力层，（2）重叠补丁嵌入，以及（3）卷积前馈网络。通过这些改进，PVT v2将PVT v1的计算复杂度降低到线性水平，并在分类、检测和分段等基础视觉任务上取得了显著改进。值得注意的是，所提议的PVT v2性能与近期如Swin Transformer相当甚至更好。

- 论文链接：[[2106.13797\]]PVT v2: Improved Baselines with Pyramid Vision Transformer(https://arxiv.org/abs/2106.13797)
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
PVT-V2 (Improved Pyramid Vision Transformer)使用 ImageNet数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/PVT-V2/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   torchrun --nproc_per_node=${NUM_PROC} train.py \
    --data-dir /data/teco-data/imagenet \
    --model pvt_v2_b0 \
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

MeanRelativeError: -8.255208774228661e-05
MeanAbsoluteError: -0.0008352251336126045
Rule,mean_absolute_error -0.0008352251336126045
pass mean_relative_error=-8.255208774228661e-05 <= 0.05 or mean_absolute_error=-0.0008352251336126045 <= 0.0002
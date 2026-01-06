# RegNetZ
## 1. 模型概述
RegNetZ，通过扩展基础卷积网络，赋予其更高的计算复杂度和相应的表示能力的过程。缩放策略的示例可能包括增加模型宽度、深度、分辨率等。虽然存在多种规模化策略，但其权衡尚未完全理解。现有分析通常侧重于准确率与浮点运算（flops）之间的相互作用。然而，正如我们所展示的，不同的缩放策略对模型参数、激活以及实际运行时间的影响截然不同。在RegNetZ中，多种缩放策略能够产生相似准确但性质差异很大的网络。这促使研究员提出了一种简单的快速复合缩放策略，主要鼓励模型宽度缩放，同时在较小程度上增加深度和分辨率。

- 论文链接：[[2103.06877\]]Fast and Accurate Model Scaling(https://arxiv.org/abs/2103.06877)
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
RegNetZ使用 ImageNet数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/RegNetZ/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   torchrun --nproc_per_node=${NUM_PROC} train.py \
    --data-dir /data/teco-data/imagenet \
    --model regnetz_e8 \
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

MeanRelativeError: 8.77262801896025e-05
MeanAbsoluteError: 0.0005228967949895576
Rule,mean_relative_error 8.77262801896025e-05
pass mean_relative_error=8.77262801896025e-05 <= 0.05 or mean_absolute_error=0.0005228967949895576 <= 0.0002
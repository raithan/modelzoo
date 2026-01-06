# PNasNet
## 1. 模型概述
PNasNet，提出了一种学习卷积神经网络（CNN）结构的新方法，其效率优于基于强化学习和进化算法的最新先进方法。我们的方法采用顺序模型优化（SMBO）策略，按复杂度递增的顺序搜索结构，同时学习替代模型以引导结构空间的搜索。在同一搜索空间下直接比较显示，我们的方法在评估模型数量上比Zoph等人（2018）的强化学习方法高效多达5倍，总计算效率高出8倍。我们通过这种方式发现的结构实现了CIFAR-10和ImageNet上最先进的分类精度。

- 论文链接：[[1712.00559\]]Progressive Neural Architecture Search(https://arxiv.org/abs/1712.00559)
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
PNasNet使用 ImageNet数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/PNasNet/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   torchrun --nproc_per_node=${NUM_PROC} train.py \
    --data-dir /data/teco-data/imagenet \
    --model pnasnet5large \
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

MeanRelativeError: -1.6978528036848447e-05
MeanAbsoluteError: -8.336152180586711e-05
Rule,mean_absolute_error -8.336152180586711e-05
pass mean_relative_error=-1.6978528036848447e-05 <= 0.05 or mean_absolute_error=-8.336152180586711e-05 <= 0.0002
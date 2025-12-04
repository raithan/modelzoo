# InceptionNeXt
## 1. 模型概述
InceptionNeXt，提出将大核深度卷积分解为沿通道维度的四个平行分支，即小方形核、两个正交带核和恒等映射。通过这一新的《盗梦空间深度卷积》，我们构建了一系列网络，即IncepitonNeXt，这些网络不仅拥有高吞吐量，还保持了竞争力的性能。

- 论文链接：[[2303.16900\]]InceptionNeXt: When Inception Meets ConvNeXt(https://arxiv.org/abs/2303.16900)
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
InceptionNeXt使用 ImageNet数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/InceptionNeXt/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   torchrun --nproc_per_node=${NUM_PROC} train.py \
    --data-dir /data/teco-data/imagenet \
    --model inception_next_small \
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

MeanRelativeError: -0.0005847310909265533
MeanAbsoluteError: -0.009622063967260983
Rule,mean_absolute_error -0.009622063967260983
pass mean_relative_error=-0.0005847310909265533 <= 0.05 or mean_absolute_error=-0.009622063967260983 <= 0.0002
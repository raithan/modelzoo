# MViT-V2 (Improved Multiscale Vision Transformer)
## 1. 模型概述
MViT-V2 (Improved Multiscale Vision Transformer) 将多尺度视觉转换器（MViTv2）作为图像和视频分类以及物体检测的统一架构进行研究。我们提出了改进版的MViT，包含分解的相对位置嵌入和残差池连接。我们将该架构实例化为五种尺寸，并评估其在ImageNet分类、COCO检测和Kinetics视频识别方面，表现优于以往工作。我们还进一步比较了 MViTv2 的注意力集中与窗口注意力机制，在准确性和计算上优于后者。MViTv2在三个领域均具备最先进的性能：ImageNet分类准确率为88.8%，在COCO对象检测中为58.7%的boxAP，以及在Kinetics-400视频分类方面为86.1%。

- 论文链接：[[2112.01526\]]MViTv2: Improved Multiscale Vision Transformers for Classification and Detection(https://arxiv.org/abs/2112.01526)
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
MViT-V2 (Improved Multiscale Vision Transformer)使用 ImageNet数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib /Classification/MViT-V2/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   torchrun --nproc_per_node=${NUM_PROC} train.py \
    --data-dir /data/teco-data/imagenet \
    --model mvitv2_base \
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

MeanRelativeError: 0.00015626826192445282
MeanAbsoluteError: 0.0011033681359621558
Rule,mean_relative_error 0.00015626826192445282
pass mean_relative_error=0.00015626826192445282 <= 0.05 or mean_absolute_error=0.0011033681359621558 <= 0.0002
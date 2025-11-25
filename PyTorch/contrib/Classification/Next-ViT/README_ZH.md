# Next-ViT
## 1. 模型概述
Next-ViT，开发了下卷积块（NCB）和下一个变换器块（NTB），以捕捉局部和全局信息，采用部署友好机制。随后，下一代混合策略（NHS）设计用于高效地将NCB和NTB叠加成，提升下游任务的性能。大量实验表明，Next-ViT在各种视觉任务中，在延迟/准确性权衡方面显著优于现有的CNN、ViT和CNN-Transformer混合架构。在TensorRT上，Next-ViT在COCO检测方面比ResNet高出5.5 mAP（从40.4提升到45.9），在ADE20K分段下比ResNet高出7.7%的mIoU（从38.8%提升到46.5%），延迟相近。同时，它的性能与CSWin相当，推理速度提升了3.6倍。在CoreML上，Next-ViT在COCO检测方面比EfficientFormal高出4.6 mAP（从42.6提升到47.2），在ADE20K分段方面比EfficientFormer高出3.5 mIoU（从45.1%提升到48.6%），且延迟相似。

- 论文链接：[[2207.05501\]]Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios(https://arxiv.org/abs/2207.05501)
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
Next-ViT 使用ImageNet数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/Next-ViT/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   torchrun --nproc_per_node=${NUM_PROC} train.py \
    --data-dir /data/teco-data/imagenet \
    --model nextvit_base \
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

MeanRelativeError: 0.0005980886827366313
MeanAbsoluteError: 0.0029405320044791346
Rule,mean_relative_error 0.0005980886827366313
pass mean_relative_error=0.0005980886827366313 <= 0.05 or mean_absolute_error=0.0029405320044791346 <= 0.0002
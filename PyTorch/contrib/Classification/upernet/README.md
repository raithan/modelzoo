
# Upernet
## 1. 模型概述
人类在多个层面上认知视觉世界：能够轻松地对场景进行分类并检测其中的物体，同时还能识别物体的纹理和表面及其不同的组成部分。研究了一项名为“统一感知解析”的新任务，该任务要求机器视觉系统从给定图像中识别尽可能多的视觉概念。UPerNet是一种多任务框架和一种训练策略，用于从异构图像标注中学习。进行了基准测试，并表明它能够有效地从图像中分割出各种概念。经过训练的网络将进一步应用于发现自然场景中的视觉知识。

- 论文链接：[1807.10221\]Unified Perceptual Parsing for Scene Understanding(https://arxiv.org/pdf/1807.10221)
- 仓库链接：https://github.com/open-mmlab/mmsegmentation/tree/main/configs/upernet

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
 使用 Cityspaces 数据集，该数据集为开源数据集，可从 (https://opendatalab.com/) 下载。

#### 2.2.2 处理数据集
具体配置方式可参考：https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/advanced_guides/datasets.md。


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip3 install  -U openmim 
    pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
    pip3 install opencv_python mmcv --no-deps
    mim install -e .
    pip install -r requirements.txt

    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/upernet/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
python run_upernet.py --config ../configs/upernet/upernet_r50_4xb2-80k_cityscapes-512x1024.py \
       --launcher pytorch --nproc-per-node 1 --amp 2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

![loss](./image/loss.jpg)

MeanRelativeErr0r:0.11623938402642707
MeanAbsoluteError:0.08640928492687716
Rule,mean_absolute_error0.08640928492687716
fail mean_relative_error=0.11623938402642707 < = 0.05 or mean_absolute_error=0.08640928492687716<=0.0002



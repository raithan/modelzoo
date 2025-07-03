
# Simmim
## 1. 模型概述
SimMIM，一个用于蒙版图像建模的简单框架，无需进行诸如分块蒙版和通过离散 VAE 或聚类进行标记化等特殊设计。使用 ViT-B，法通过在该数据集上进行预训练，在 ImageNet-1K 数据集上实现了 83.8% 的 top-1 微调准确率，比之前的最佳方法高出 0.6%。当将其应用于拥有约 6.5 亿个参数的更大规模模型 SwinV2H 时，仅使用 ImageNet-1K 数据，即可在 ImageNet-1K 上实现 87.1% 的 Top-1 准确率。

- 论文链接：[2111.09886\]SimMIM: A Simple Framework for Masked Image Modeling(https://arxiv.org/abs/2111.09886)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/simmim

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
Deit 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

#### 2.2.2 处理数据集
具体配置方式可参考：https://blog.csdn.net/xzxg001/article/details/142465729。


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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/simmim/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
python run_simmim.py --config ../configs/simmim/simmim_swin-base-w6_8xb256-amp-coslr-100e_in1k-192px.py \
       --launcher pytorch --nproc-per-node 1 --amp \
       --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/imagenet" 2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

![loss](./image/loss.jpg)

MeanRelativeError:0.0019742962462511566
MeanAbsoluteError:0.000976111629221699
Rule,mean_absolute_error 0.000976111629221699
passmean_relative_error=0.0019742962462511566 <=0.05 or mean_absolute_error=0.000976111629221699 <=0.0002



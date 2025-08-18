
# DenseCL
## 1. 模型概述
MoCo v2（Momentum Contrast v2）是由Facebook AI Research (FAIR)团队提出的一种改进的无监督对比学习框架。其核心思想是通过在原始MoCo基础上引入MLP投影头（MLP projection head）和增强的数据扩充策略两项关键改进，在保持原有动量对比机制优势的同时，显著提升了特征表示的质量和迁移能力。相较于原始MoCo框架，MoCo v2通过简单的结构修改：1）采用多层感知机替代线性投影层，增强特征非线性表达能力；2）引入更丰富的数据增强策略优化正样本构建，有效解决了SimCLR方法依赖超大训练批次（large training batches）的限制。该方法在不需要大训练批次的条件下，不仅超越了同期SimCLR的性能表现，还成为自监督对比学习领域的新基准模型，为无监督学习研究提供了更高效、更易复现的技术路径。



- 论文链接：[[2003.04297\]]Improved Baselines with Momentum Contrastive Learning(https://arxiv.org/abs/2003.04297)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/mocov2

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
densecl 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
  cd <ModelZoo_path>/PyTorch/contrib/Classification/mocov2/run_scripts
  ```

2. 运行训练。该模型支持单机单卡。
  ```
  python run_mocov2.py --config ../configs/mocov2/benchmarks/resnet50_8xb32-linear-steplr-100e_in1k.py --launcher pytorch --nproc-per-node 4 --amp --cfg-options "train_dataloader.dataset.data_root=<imagenet_path>" "val_dataloader.dataset.data_root=<imagenet_path>"
  ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果

|模型             |    数据集      |    sdaa结果       |   cuda结果         |  sdaa耗时    |    
|:---------------|:--------------:|:-----------------:|:-----------------:|:-------------:|
|mocov2    |imagenet-1k  |67.5420     |  67.50            |2d         |
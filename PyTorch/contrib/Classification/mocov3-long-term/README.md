# mocov3
## 1. 模型概述
MoCov3 是一种用于视觉表征学习的自监督学习方法，由He等人团队在2021年提出。它并非提出全新的框架，而是对已有的对比学习方法进行了改进和系统化研究，特别是针对 Vision Transformer (ViT) 的自监督训练问题。相比于卷积神经网络已经较为成熟的训练范式，ViT 在自监督场景下的训练更加不稳定，MoCov3的核心贡献在于回归基础要素，探索并优化训练过程中的关键组件。MoCov3的主要发现是：训练不稳定性是影响ViT自监督性能的关键瓶颈。研究表明，一些表面上看似良好的结果实际上隐藏了部分失败，而通过改善训练稳定性，模型表现可以得到显著提升。在方法层面，MoCov3采用了更简洁直接的训练策略，结合对比学习框架，对ViT进行了广泛的实验和消融分析，验证了自监督学习在Transformer结构中的可行性和潜力。总体而言，MoCov3为后续研究提供了宝贵的经验和基线，它不仅揭示了自监督ViT训练中的挑战与陷阱，还提出了更稳定有效的训练实践。作为一种“必知的基线”，MoCov3在推动自监督学习与Transformer结合的研究进程中具有重要意义，并为后续视觉表示学习的发展奠定了基础。


## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境
4. 启动训练：介绍如何运行训练。

### 2.1  基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
此处mocov3使用的是 imagenet 数据集，需要下载数据集。
下载链接为https://www.image-net.org/challenges/LSVRC/index.php

#### 2.2.2 处理数据集
解压数据集，使用如下格式进行训练
```
|-imagenet
    |-train
        |-...
    |-val
        |-...
    |-train_list.txt
    |-val_copy_list.txt
```

### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
执行以下命令，启动虚拟环境。
conda activate torch_env

安装python依赖
pip install -r requirements.txt


### 2.4 启动训练
在构建好的环境中，进入训练脚本所在目录。

cd <ModelZoo_path>/PyTorch/contrib/Classification/mocov3-long-term
运行训练。

运行命令
python run_scripts/run_mocov3.py \
  --config config/resnet50_8xb128-linear-coslr-90e_in1k.py\
  --launcher pytorch --nproc-per-node 32 \
  --cfg-options \
     train_dataloader.dataset.data_root=/data/teco-data/imagenet \
     val_dataloader.dataset.data_root=/data/teco-data/imagenet \
  2>&1 | tee sdaa.log

### 2.4 训练结果
结果如下  
|加速卡数量|模型|混合精度|Epoch|Batch size|sdaa Top-1|cuda Top-1|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|8|mocov3|是|50|256|66.82|69.6|
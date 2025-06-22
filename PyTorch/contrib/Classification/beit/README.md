
# Beit
## 1. 模型概述
自监督视觉表征模型 BEiT，即双向编码器表征，它源自图像变换器 (Image Transformers)。借鉴自然语言处理领域中发展的 BERT，提出了一个带掩码的图像建模任务来预训练视觉变换器 (Vision Transformers)。在对 BEiT 进行预训练后，通过在预训练的编码器上附加任务层，直接在下游任务中微调模型参数。图像分类和语义分割的实验结果表明，这个模型取得了与以往预训练方法相当的成果。


- 论文链接：[2106.08254\]BEiT: BERT Pre-Training of Image Transformers Modeling(https://arxiv.org/abs/2106.08254)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/beit

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/beit/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
python run_beit.py --config ../configs/beit/benchmarks/beit-base-p16_8xb64_in1k.py \
       --launcher pytorch --nproc-per-node 1 --amp \
       --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/imagenet" "val_dataloader.dataset.data_root=/data/teco-data/imagenet" 2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

![loss](./image/loss.jpg)

MeanRelativeError:-0.00012557125954236584
MeanAbsoluteError:-0.001080687683407623
Rule,mean_absolute_error：-0.001080687683407623
passmean_relative_error=-0.00012557125954236584 <=0.05ormean_absolute_error=-0.001080687683407623<=0.0002



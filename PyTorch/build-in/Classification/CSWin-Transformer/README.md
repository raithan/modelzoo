# CSWin-Transformer

## 1. 模型概述
CSWin Transformer（CSWin 是 Cross-Shaped Window 的缩写）在 arXiv 上被提出，它是一种用于计算机视觉的新型通用主干网络。它是一种分层 Transformer，用我们新提出的十字形窗口自注意力机制取代了传统的全注意力机制。十字形窗口自注意力机制在水平和垂直条纹上并行计算自注意力，从而形成十字形窗口，其中每条条纹是通过将输入特征分割成等宽的条带获得的。借助 CSWin，我们能够在有限的计算成本下实现全局注意力。CSWin Transformer的代码主要从[GitHub]迁移和调整 [GitHub](https://github.com/microsoft/CSWin-Transformer).

- 仓库链接：[pretrained-models.pytorch](https://github.com/microsoft/CSWin-Transformer)

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
CSWin Transformer运行在ImageNet上，这是一个来自ILSVRC挑战赛的广受欢迎的图像分类数据集。您可以点击[此链接](https://image-net.org/download-images)从公开网站中下载数据集。

#### 2.2.2 处理数据集
· 执行以下命令，解压训练数据集。
```
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

```
· 执行以下命令，解压测试数据并将图像移动到子文件夹中。
```
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

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
3. 安装已经适配sdaa的timm==1.0.15。
    ```
    pip install timm-1.0.15-py3-none-any.whl
    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/build-in/Classification/SENet154
    ```

2. 运行训练。

    单核组
    ```
    python main.py --data /data/dataset/imagenet --model CSWin_64_12211_tiny_224 -b 16 --lr 0.25e-3 --weight-decay .05 --amp --img-size 224 --warmup-epochs 1 --model-ema-decay 0.99984 --drop-path 0.2
    ```

    单机单卡

    ```
   python -m torch.distributed.launch --master_port 50130 --nproc_per_node=4 --use_env main.py --data /data/dataset/imagenet --model CSWin_64_12211_tiny_224 -b 16 --lr 0.25e-3 --weight-decay .05 --amp --img-size 224 --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.2
    ```
    更多训练参数参考(参考使用[README_EN.md](./README_EN.md))

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./loss.py)）: 
MeanRelativeError: 0.0007178764089637323
MeanAbsoluteError: 0.004742999999999977
Rule,mean_relative_error 0.0007178764089637323
pass mean_relative_error=0.0007178764089637323 <= 0.05 or mean_absolute_error=0.004742999999999977 <= 0.0002


pass mean_relative_error=np.float64(0.014464888744248195) <= 0.05 or mean_absolute_error=np.float64(0.054114364996189025) <= 0.0002


# ResNet

## 1. 模型概述
ResNet（Residual Neural Network）是由微软研究院提出的一种深度残差网络，通过引入残差结构解决了深层神经网络训练中的梯度消失和退化问题。

本项目基于 ResNet 模型结构，使用 CIFAR-10 数据集进行图像分类实验。

- 论文链接：[1512.03385](https://arxiv.org/abs/1512.03385)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/configs/resnet


## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装
2. 获取数据集
3. 构建环境
4. 启动训练

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
### 2.2.1 准备数据集
Resnet 使用 CIFAR-10 数据集，该数据集为开源数据集，可从 [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 下载。

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
cd <ModelZoo_path>/PyTorch/contrib/Classification/resnet_cifar/run_scripts
```

2. 运行训练。该模型支持单机单卡
```
python run_resnet.py --config ../configs/resnet18_8xb16_cifar10.py --launcher pytorch --nproc-per-node 4 --amp --cfg-options "train_dataloader.dataset.data_root=<cifar10_path>" "val_dataloader.dataset.data_root=<cifar10_path>"
```

### 2.5 训练结果
| Model                     | Pretrain     | Params (M) | Flops (G) | Top-1 (%) sdaa | Top-1 (%) cuda | Total Time |
|---------------------------|--------------|------------|-----------|----------------|----------------|------------|
| `resnet18_8xb16_cifar10`  | From scratch | 11.17      | 0.56      | 94.77          | 94.82          | 90min      |

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

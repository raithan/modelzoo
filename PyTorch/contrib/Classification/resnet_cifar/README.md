# ResNet

## 1. Model Overview
ResNet (Residual Neural Network) is a deep residual network architecture proposed by Microsoft Research. It addresses the vanishing gradient and degradation problems in deep neural networks through innovative residual connections.
This project implements the ResNet architecture for image classification on the CIFAR-10 dataset.
- Paper:[1512.03385](https://arxiv.org/abs/1512.03385)
- Code Repository:https://github.com/open-mmlab/mmpretrain/configs/resnet


## 2. Quick Start

### 2.1 Prerequisites

Before beginning, ensure your system meets the basic requirements outlined in the Environment Setup section.

### 2.2 Dataset Preparation
### 2.2.1 Download CIFAR-10 Dataset
The ResNet model uses the CIFAR-10 dataset, which is an open-source dataset available for download at [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 

### 2.3 Environment Setup
The following steps assume you have a PyTorch-compatible environment
1. Activate your virtual environment
```
conda activate torch_env
```

2. Install Python dependencies
```
pip3 install  -U openmim 
pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
pip3 install opencv_python mmcv --no-deps
mim install -e .
pip install -r requirements.txt
```
### 2.4 Training Execution
1. Navigate to the training script directory
```
cd <ModelZoo_path>/PyTorch/contrib/Classification/resnet_cifar/run_scripts
```

2. Start training (supports single-node single-GPU configuration):
```
python run_resnet.py --config ../configs/resnet18_8xb16_cifar10.py --launcher pytorch --nproc-per-node 4 --amp --cfg-options "train_dataloader.dataset.data_root=<cifar10_path>" "val_dataloader.dataset.data_root=<cifar10_path>"
```

### 2.5 Performance Results
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

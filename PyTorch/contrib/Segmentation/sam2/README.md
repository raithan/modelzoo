
# SAM
## 1. 模型概述
最新的SAM2分割大模型（Segment Anything Model 2）是由Meta开发的一个先进的图像和视频分割模型。相比于第一代SAM模型，SAM2在多个方面实现了显著的改进：
1. 支持视频分割：SAM2的一个重要进展是它的能力从图像分割扩展到了视频分割。这意味着它能够处理视频中的对象，而不仅仅是静态图像。
2. 实时处理任意长视频：SAM2能够实时处理任意长度的视频，这在实际应用中非常有用，尤其是在需要快速响应的场景中。
3. Zero-shot泛化：即使是在视频中没有见过的对象，SAM2也能实现有效的分割和追踪，这显示了其强大的泛化能力。
原始仓库链接：https://github.com/facebookresearch/sam2。

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集和权重
#### 2.2.1 获取数据集
> 我们在本项目中使用了 LabPicsV1 数据集。请把数据集下载到sam2文件夹中。数据集下载链接：https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1


#### 2.2.2 处理数据集
> 解压训练数据集：
```
unzip LabPicsV1.zip
```

#### 2.2.3 数据集目录结构

数据集目录结构参考如下所示:
```
## LabPicsV1
├── Complex
│   ├── PythonReaders
│   ├── Test
│   └── Train
├── EvaluationScriptsPython
├── Other
└── Simple
```

#### 2.2.4 获取SAM2权重
> 执行以下命令：
```
cd /data/bigc-data/ltb/sam2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
    ```
    conda create -n sam2 --clone torch_env
    conda activate sam2
    ```

2. 安装python依赖
    ```
    pip install -r requirements.txt
    cd /PyTorch/contrib/Segmentation/sam2
    pip install -e .
    ```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd /PyTorch/contrib/Segmentation/sam2
    ```

2. 运行训练。该模型支持单机单卡.

    -  单机单卡
    ```
   python train.py 2>&1 | run_scripts/tee sdaa.log
    ```

    

### 2.5 训练结果w
训练loss曲线: 
![训练loss曲线](./run_scripts/loss.png)

最小 loss: 0.01656183786690235
最大 loss: 0.5936434268951416

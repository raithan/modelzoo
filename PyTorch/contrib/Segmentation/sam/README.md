
# SAM
## 1. 模型概述
Segment Anything Model （SAM） 根据输入提示（例如点或框）生成高质量的对象掩码，并且可用于为图像中的所有对象生成掩码。它已经在包含 1100 万张图像和 11 亿个掩码的数据集上进行了训练，在各种分割任务上具有很强的零样本性能
原始仓库链接：https://github.com/facebookresearch/segment-anything。
微调代码链接：https://github.com/xzyun2011/finetune_segment_anything_tutorial

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
> 我们在本项目中使用了 VOC2007  数据集。请把数据集下载到data_example文件夹中链接：http://host.robots.ox.ac.uk/pascal/VOC/voc2007/


#### 2.2.2 处理数据集
> 解压训练数据集：
```
tar -xvf VOCtrainval_06-Nov-2007-1.tar
```

#### 2.2.3 数据集目录结构

数据集目录结构参考如下所示:
```
## VOCdevkit/VOC2007
├── Annotations
├── ImageSets
│   ├── Layout
│   ├── Main
│   └── Segmentation
├── JPEGImages
├── SegmentationClass
└── SegmentationObject
```

#### 2.2.4 获取SAM权重
> 执行以下命令：
```
cd /data/bigc-data/ltb/SAM/weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
    ```
    conda create -n SAM --clone torch_env
    conda activate SAM
    ```

2. 安装python依赖
    ```
    pip install -r requirements.txt
    cd /PyTorch/contrib/Segmentation/SAM/segment-anything-main
    pip install -e .
    ```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd /PyTorch/contrib/Segmentation/SAM/run_scripts
    ```

2. 运行训练。该模型支持单机单卡.

    -  单机单卡
    ```
   sh ./run_scripts/train.sh ./run_scripts/run_sam.py 2>&1 | tee run_scripts/sdaa.log
    ```

    

### 2.5 训练结果w
训练loss曲线: 
![训练loss曲线](./run_scripts/loss.png)

最小 loss: 0.03177395835518837
最终 loss: 3.7017173767089844

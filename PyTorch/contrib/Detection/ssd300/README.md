
# SSD300
## 1. 模型概述
SSD300（Single Shot MultiBox Detector）是由 Google 在 2016 年提出的一种端到端单阶段目标检测网络。其核心思想是通过多尺度特征图同时进行预测，兼顾检测速度与精度，在保证实时性的同时取得较好性能。SSD300 在 VOC 和 COCO 数据集上表现优异，成为轻量高效目标检测模型的经典代表。

- 论文链接：[[1512.02325]] SSD: Single Shot MultiBox Detector (https://arxiv.org/abs/1512.02325)
- 仓库链接：https://github.com/open-mmlab/mmdetection/tree/main/configs/ssd

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
SSD300 常用 COCO 数据集：COCO 数据集可从 COCO官网 下载。

#### 2.2.2 处理数据集
具体配置方式可参考：https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html


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
  cd <ModelZoo_path>/PyTorch/contrib/Detection/ssd300/run_scripts
  ```

2. 运行训练。该模型支持单机单卡。
  ```
  python run_ssd300.py --config ../configs/ssd300/ssd300_coco.py --launcher pytorch --nproc-per-node 4 --amp --cfg-options "train_dataloader.dataset.data_root=<coco_path>" "val_dataloader.dataset.data_root=<coco_path>"

  ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
 2.5 训练结果

|      模型       |    数据集      |    sdaa结果       |   cuda结果         |  sdaa耗时    |    
|:---------------|:--------------:|:-----------------:|:-----------------:|:-------------:|
|ssd300    |coco  |25.3     |  25.5            |1d22h         |
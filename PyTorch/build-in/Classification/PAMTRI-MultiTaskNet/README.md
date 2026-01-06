# MultiTaskNet

## 1. 模型概述
本目录包含了论文 PAMTRI: 使用高度随机化合成数据的姿态感知多任务学习用于车辆重识别（ICCV 2019）中多任务网络的官方 PyTorch 实现，主要对PAMTRI中MultiTaskNet进行部分修改。
- 仓库链接：[PAMTRI](https://github.com/NVlabs/PAMTRI/tree/master)
- 更多信息参考readme_en.md

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
VeRi数据集：[VeRi](https://vehiclereid.github.io/VeRi/)

#### 2.2.2 处理数据集
· 数据集目录结构如下
```
${REID_ROOT}
 `-- data
     |-- veri
         |-- image_query
         |   |-- 0002_c002_00030600_0.jpg
         |   |-- 0002_c003_00084280_0.jpg
         |   |-- ...
         |-- image_test
         |   |-- 0002_c002_00030600_0.jpg
         |   |-- 0002_c002_00030605_1.jpg
         |   |-- ...
         |-- image_train
         |   |-- 0001_c001_00016450_0.jpg
         |   |-- 0001_c001_00016460_0.jpg
         |   |-- ...
         |-- heatmap_query
         |   |-- 0002_c002_00030600_0
         |   |   |-- 00.jpg
         |   |   |-- 01.jpg
         |   |   |-- ...
         |   |   `-- 35.jpg
         |   |-- ...
         |-- heatmap_test
         |   |-- 0002_c002_00030600_0
         |   |   |-- 00.jpg
         |   |   |-- 01.jpg
         |   |   |-- ...
         |   |   `-- 35.jpg
         |   |-- ...
         |-- heatmap_train
         |   |-- 0001_c001_00016450_0
         |   |   |-- 00.jpg
         |   |   |-- 01.jpg
         |   |   |-- ...
         |   |   `-- 35.jpg
         |   |-- ...
         |-- segment_query
         |   |-- 0002_c002_00030600_0
         |   |   |-- 00.jpg
         |   |   |-- 01.jpg
         |   |   |-- ...
         |   |   `-- 12.jpg
         |   |-- ...
         |-- segment_test
         |   |-- 0002_c002_00030600_0
         |   |   |-- 00.jpg
         |   |   |-- 01.jpg
         |   |   |-- ...
         |   |   `-- 12.jpg
         |   |-- ...
         |-- segment_train
         |   |-- 0001_c001_00016450_0
         |   |   |-- 00.jpg
         |   |   |-- 01.jpg
         |   |   |-- ...
         |   |   `-- 12.jpg
         |   |-- ...
         |-- label_query.csv
         |-- label_test.csv
         `-- label_train.csv
     |-- ...
具体数据集配置参考readme_en.md,需要放置在data下
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

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/build-in/Classification/PAMTRI-MultiTaskNet
    ```

2. 运行训练。该模型支持单机单核组。

    ```
    export TORCH_SDAA_AUTOLOAD=cuda_migrate  #自动迁移环境变量
    python train.py -d veri -a densenet121 --root data --save-dir log/densenet121-xent-htri-veri-multitask --gpu-devices 0  --step 100 
    ```

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./loss.py)）: 





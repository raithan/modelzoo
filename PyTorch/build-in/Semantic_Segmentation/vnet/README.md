# vnet

## 1. 模型概述
VNet 是一种专为医学三维图像分割设计的全卷积神经网络，它采用编码器-解码器结构，通过直接学习输入与输出之间的端到端映射，实现对如 MRI 或 CT 体积数据的精确分割。该模型借鉴了 U-Net 的跳跃连接思想，在下采样（编码）过程中保留深层语义信息的同时，通过上采样（解码）和特征融合恢复空间细节，特别适用于处理对比度低、边界模糊的医学影像。V-Net 的损失函数通常基于 Dice 系数，能有效应对前景与背景像素极度不平衡的问题，广泛应用于器官、肿瘤等医学区域的自动分割任务。

- 参考实现：
    ```
    url=https://github.com/mattmacy/vnet.pytorch
    commit_id=a00c8ea16bcaea2bddf73b2bf506796f70077687
    ```


## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 准备数据集

- 训练Vnet需要使用到开源数据集LUNA16，请将数据集上传到服务器任意路径下并解压。

- 数据集目录结构参考如下所示：

   ```
    ├── LUNA16
        ├──lung_ct_image
            ├──1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335.mhd                    
            ├──...                     
        ├──seg-lungs-LUNA16
            ├──1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335.mhd
            ├──...
        ├──normalized_lung_ct
            ├──1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335.mhd                    
            ├──...                     
        ├──normalized_lung_mask
            ├──1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335.mhd
            ├──...    
   ```
数据集生成说明：
将下载的原始数据集中的所有文件移动到lung_ct_image目录下，执行以下命令得到normalized_lung_ct以及normalized_lung_mask两个文件夹
    ```
    python normalize_dataset.py xxx/vnet/luna16/ 2.5 128 160 160
    ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Semantic_Segmentation/vent
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_vent`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_vent
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_vent -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_vent /bin/bash
    ```

#### 2.3.5 执行以下命令，启动虚拟环境。

    ```
    conda activate torch_env_py310
    ```

#### 2.3.6 执行以下命令，安装其他环境依赖包。

    ```
    pip install -r requirements.txt
    ```

#### 2.3.7 执行以下命令，从源码安装torchbiomed。
    ```
    git clone https://github.com/mattmacy/torchbiomed.git
    cd torchbiomed
    pip install -e .
    ```

### 2.4 启动训练

#### 2.4.1 在Docker环境中，进入训练脚本所在目录。
    ```
    cd /workspace/Semantic_Segmentation/vent
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
     python  train.py
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [0.7665 0.6919 0.6314 0.6475 0.5828 0.5176 0.4859 0.496  0.4475 0.4246]
Parsed loss array (first 10): [0.8322 0.7015 0.6935 0.6992 0.6458 0.569  0.4551 0.4441 0.4019 0.375 ]
MeanRelativeError: -0.21686137
MeanAbsoluteError: -0.061338678
Rule,mean_relative_error -0.21686137
pass mean_relative_error=-0.21686137 <= 0.05 or mean_absolute_error=-0.061338678 <= 0.0002



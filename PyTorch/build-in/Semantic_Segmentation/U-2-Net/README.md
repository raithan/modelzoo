# U-2-Net

## 1. 模型概述
U-2-Net 是一种基于嵌套 U-Net 结构的高效显著性目标检测与语义分割模型，采用双级嵌套的编码器-解码器架构（ReSidual U-blocks），能够在无背景信息的情况下精准分割图像中的主要物体。该模型通过多尺度特征融合和深层监督机制，在保持轻量化的同时实现了高精度的边缘细节保留，广泛应用于图像抠图、前景提取、医学图像分割等任务，尤其适合需要高质量像素级预测的场景。


- 参考实现：
    ```
    url=https://github.com/xuebinqin/U-2-Net
    commit_id=ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29
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

- 使用的是DUTS-TR，它是DUTS数据集的一部分。DUTS-TR一共包含10553张图像。目前，它是用于显著性目标检测的最大和最常用的训练数据集。通过水平翻转来扩充数据集，共获得21106个训练图像。将数据集上传到服务器任意路径下并解压，需要对应的修改u2net_train.py脚本中的数据集路径。

- 若直接使用仓库中的训练脚本进行训练，请按以下结构组织数据集：

   ```
    ├── train_data
    │    ├──DUTS-TR
    │    │      │  DUTS-TR-Image
    │    │      │      │  ...     
    │    │      │  DUTS-TR-Mask    
    │    │      │      │  ...     
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Semantic_Segmentation/U-2-Net
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_U-2-Net`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_U-2-Net
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_U-2-Net -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_U-2-Net /bin/bash
    ```

#### 2.3.5 执行以下命令，启动虚拟环境。

    ```
    conda activate torch_env_py310
    ```

#### 2.3.6 执行以下命令，安装其他环境依赖包。

    ```
    pip install -r requirements.txt
    ```


### 2.4 启动训练

#### 2.4.1 在Docker环境中，进入训练脚本所在目录。
    ```
    cd /workspace/Semantic_Segmentation/U-2-Net
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    python u2net_train.py
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [5.008827 4.731226 6.951201 8.139488 9.364314 8.713507 8.44922  8.102379
 7.77872  7.397047]
Parsed loss array (first 10): [5.045554 9.939081 9.513636 8.591209 8.02963  7.486432 7.125776 6.810367
 6.573476 6.40521 ]
MeanRelativeError: 0.030790253
MeanAbsoluteError: 0.14943928
Rule,mean_relative_error 0.030790253
pass mean_relative_error=0.030790253 <= 0.05 or mean_absolute_error=0.14943928 <= 0.0002



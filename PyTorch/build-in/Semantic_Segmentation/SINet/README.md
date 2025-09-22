# SINet

## 1. 模型概述
SINet（Shallow Interaction Network）是一种专为显著性目标检测（尤其是复杂背景下的小物体检测）设计的轻量级语义分割模型。它通过引入浅层-深层特征交互机制（Shallow-Deep Interaction）和边界感知损失函数，有效增强了网络对微小目标的敏感性与边缘细节的刻画能力。SINet 采用分组卷积与注意力门控模块，在保证实时推理速度的同时，实现了高精度的显著性区域分割，特别适用于无人机图像、遥感影像和监控视频中的小目标识别与提取任务。


- 参考实现：
    ```
    url=https://github.com/DengPingFan/SINet
    commit_id=6202fb10efb6a36b36ebb3c0a251fd4360a6c76a
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

- 请你从[谷歌网盘](https://drive.google.com/file/d/1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt/view?usp=sharing)下载COD10K-train数据集，将数据集上传到服务器任意路径下并解压。

- 请你按以下结构组织COD10K数据集：Dataset/TrainDataset/

   ```
    ├── Dataset
    │    ├──TrainDataset
    │    │      │  Edge      
    │    │      │  GT
    │    │      │  Image        
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Semantic_Segmentation/SINet
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_SINet`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_SINet
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_SINet -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_SINet /bin/bash
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
    cd /workspace/Semantic_Segmentation/SINet
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    python MyTrain.py
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [0.6958 0.6641 0.6678 0.6573 0.6502 0.6422 0.6375 0.6144 0.5984 0.6049]
Parsed loss array (first 10): [0.6857 0.6666 0.6455 0.6508 0.6206 0.6067 0.6094 0.5927 0.6077 0.6395]
MeanRelativeError: 0.02367913
MeanAbsoluteError: 0.0058961157
Rule,mean_absolute_error 0.0058961157
pass mean_relative_error=0.02367913 <= 0.05 or mean_absolute_error=0.0058961157 <= 0.0002


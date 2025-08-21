# SiamFC

## 1. 模型概述
SiamFC（Fully-Convolutional Siamese Network）是一种用于视频目标跟踪的经典深度学习模型，它采用全卷积孪生网络结构，通过离线训练学习一个通用的相似性匹配函数。在跟踪过程中，模型将第一帧中目标模板与后续帧的搜索区域进行卷积匹配，生成响应图以预测目标位置，实现端到端的高效跟踪。SiamFC的优势在于速度较快、无需在线微调，且通过离线训练即可实现良好的泛化能力，为后续大量基于孪生网络的跟踪算法奠定了基础。


- 参考实现：
    ```
    url=https://github.com/HonglinChu/SiamTrackers/tree/master/SiamFC/SiamFC
    commit_id=2dd15d2591d8f34074b3074c0680fbc962c40cc6
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

- 用户自行获取原始数据集，下载GOT-10K数据集，将数据集上传到服务器任意路径下并解压。

- 请你按以下结构组织GOT-10K数据集：
   ```
    GOT-10k/
    ├── train/
    │   ├── GOT-10k_Train_000001/
    │   │   ├── 000000.jpg
    │   │   ├── 000001.jpg
    │   │   └── 000002.jpg
    │   ├── GOT-10k_Train_000002/
    │   │   ├── 000000.jpg
    │   │   ├── 000001.jpg
    │   │   └── 000002.jpg
    ├── val/
    │   ├── GOT-10k_Val_000001/
    │   │   ├── 000000.jpg
    │   │   ├── 000001.jpg
    │   └── GOT-10k_Val_000002/
    │       ├── 000000.jpg
    ├── test/
    │   ├── GOT-10k_Test_000001/
    │   │   ├── 000000.jpg
    │   └── GOT-10k_Test_000002/
    ├── GOT-10k_Train_Meta/
    │   ├── meta_archive.json
    │   └── train_labels.csv
    ├── GOT-10k_val.csv
    ├── GOT-10k_train.csv
    └── GOT-10k_test.csv
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Video/SiamFC
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_SiamFC`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_SiamFC
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_SiamFC -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_SiamFC /bin/bash
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
    cd /workspace/Video/SiamFC
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    python ./bin/my_train.py
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: 0.0014232274
MeanAbsoluteError: 0.0010527384
Rule,mean_absolute_error 0.0010527384
pass mean_relative_error=0.0014232274 <= 0.05 or mean_absolute_error=0.0010527384 <= 0.0002


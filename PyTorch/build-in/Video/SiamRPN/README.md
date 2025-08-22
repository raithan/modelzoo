# SiamRPN

## 1. 模型概述
SiamRPN（Fully-Convolutional Siamese Region Proposal Network）是一种结合孪生网络与区域建议网络（RPN）的高效视频目标跟踪模型，它通过离线训练一个通用的相似性匹配网络，将模板图像和搜索区域分别输入共享权重的孪生子网络提取特征，再利用RPN在特征层面进行密集的分类与回归，实现在不进行在线微调的情况下快速生成目标候选框。该方法继承了SiamFC的速度优势，同时通过引入RPN结构显著提升了对目标尺度变化和位置偏移的适应能力，是经典的相关滤波和孪生网络跟踪方法的重要演进之一


- 参考实现：
    ```
    url=https://github.com/STVIR/pysot
    commit_id=9b07c521fd370ba38d35f35f76b275156564a681
    ```


## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 准备数据集以及预训练权重

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
- 请你从[此链接](https://cas-bridge.xethub.hf.co/xet-bridge-us/686ca38bc3516f3578fa6a75/6128aabd0c61e7a9c99773eed584bf9623a7574ea0e0de72fb5731d8cfbce349?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20250730%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250730T054658Z&X-Amz-Expires=3600&X-Amz-Signature=6d13bb276863c8e0688acadbdab3af3666d74fe226ec1d34f1f2eb9ce7bb8cda&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=public&response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27alexnet_model.pth%3B+filename%3D%22alexnet_model.pth%22%3B&x-id=GetObject&Expires=1753858018&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc1Mzg1ODAxOH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82ODZjYTM4YmMzNTE2ZjM1NzhmYTZhNzUvNjEyOGFhYmQwYzYxZTdhOWM5OTc3M2VlZDU4NGJmOTYyM2E3NTc0ZWEwZTBkZTcyZmI1NzMxZDhjZmJjZTM0OSoifV19&Signature=GoI0dOieWCW2fc1Lqewz1ucED%7ETL4iy0dORNqskzwVr6qFHkHRktvhqp-8OjJiStDSQo5B65kOUK9wlVMoz9Gcd6NZnP9bQ1n4eVGkpiypPAlGSjX2JauFb3X2GKwaUy%7E7SKl-TltKpIS2%7E%7EKxV7u0D23f2k389-w3Jp9WRvNxW-cDQlIDLLU1RWQoNZp70B%7E3HHLkyDf-myNzqNSf3czZjwVKxePlLZYnjgoxyuxkc66f0N-Vla6I6x9PpDgmzKCi6qYl0pKMxF8rB07jvUpiJT1zN01OE9DJsuMY0BnLG%7EKuf17ojLNLTlQOzn-qBUwk3Yv1IGgJZNVow7EjhxDA__&Key-Pair-Id=K2L8F4GPSG1IFC)下载pth文件，并放置在models路径下。

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Video/SiamRPN
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_SiamRPN`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_SiamRPN
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_SiamRPN -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_SiamRPN /bin/bash
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
    python ./bin/train.py
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: -0.026594782
MeanAbsoluteError: -3.963009
Rule,mean_absolute_error -3.963009
pass mean_relative_error=-0.026594782 <= 0.05 or mean_absolute_error=-3.963009 <= 0.0002


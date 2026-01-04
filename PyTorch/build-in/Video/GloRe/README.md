# GloRe

## 1. 模型概述
GloRe（Global Reasoning Network）是一种用于图像识别的神经网络模块，旨在通过建模图像中物体或区域之间的长距离依赖关系来增强卷积神经网络的全局上下文理解能力。它通过将空间特征图映射到一个低维的“语义”图空间，在该图中显式地进行全局信息聚合与交互（即“全局推理”），再将推理后的结果回传并增强原始特征，从而有效捕捉图像中分散或远距离的语义关联，提升模型在场景理解、语义分割等任务上的表现。


- 参考实现：
    ```
    url=https://github.com/facebookresearch/GloRe
    commit_id=9c6a7340ebb44a66a3bf1945094fc685fb7b730d
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

- 请用户自行获取原始数据集UCF-101，包含训练集和测试集两部分，并在模型源码包根目录./dataset/UCF101/raw/路径下新建data文件夹，将获取的数据集上传至该路径下并解压。

- 数据集目录结构参考如下所示：

   ```
    ├── ucf101
    │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
    │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
    │   ├── annotations
    │   ├── videos
    │   │   ├── ApplyEyeMakeup
    │   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi  
    │   │   ├── YoYo
    │   │   │   ├── v_YoYo_g25_c05.avi
    │   ├── rawframes
    │   │   ├── ApplyEyeMakeup
    │   │   │   ├── v_ApplyEyeMakeup_g01_c01
    │   │   │   │   ├── img_00001.jpg
    │   │   │   │   ├── img_00002.jpg
    │   │   │   │   ├── ...
    │   │   │   │   ├── flow_x_00001.jpg
    │   │   │   │   ├── flow_x_00002.jpg
    │   │   │   │   ├── ...
    │   │   │   │   ├── flow_y_00001.jpg
    │   │   │   │   ├── flow_y_00002.jpg
    │   │   ├── ...
    │   │   ├── YoYo
    │   │   │   ├── v_YoYo_g01_c01
    │   │   │   ├── ...
    │   │   │   ├── v_YoYo_g25_c05   
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Video/GloRe/
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_GloRe`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_GloRe
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_GloRe -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_GloRe /bin/bash
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
    cd /workspace/Video/GloRe/
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    python train_kinetics.py
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: 0.008593342
MeanAbsoluteError: 0.023878379
Rule,mean_relative_error 0.008593342
pass mean_relative_error=0.008593342 <= 0.05 or mean_absolute_error=0.023878379 <= 0.0002


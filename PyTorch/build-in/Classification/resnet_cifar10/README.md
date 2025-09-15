# resnet_cifar10

## 1. 模型概述
该仓库实现了专为 CIFAR-10/CIFAR-100 图像分类任务设计的 ResNet-s（Small ResNet） 模型，其结构严格遵循何凯明原版论文中针对小尺寸图像的配置，与 torchvision 中为 ImageNet 设计的标准 ResNet 不同。主要区别在于：它使用更窄的通道数、调整了初始卷积核大小，并去除了第一个池化层，以适应 32×32 的低分辨率图像，从而在参数量更少的情况下，在 CIFAR 数据集上达到论文报告的高性能（如 ResNet-56 错误率低至 6.61%），是研究 ResNet 在小数据集上表现的准确实现版本。


- 参考实现：
    ```
    url=https://github.com/akamaster/pytorch_resnet_cifar10
    commit_id=d5489e8995e81e91ce6b1d69dcc98ad579b0b153
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

- 可选用的开源数据集包括CIFAR10/CIFAR100，下载后将数据集上传到服务器任意路径下并解压。

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Classification/resnet_cifar10
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_resnet_cifar10`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_resnet_cifar10
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_resnet_cifar10 -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_resnet_cifar10 /bin/bash
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
    cd /workspace/Classification/resnet_cifar10
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    chmod 777 run.sh 
    bash ./run.sh
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [3.4358 2.5405 3.3999 3.192  3.0062 3.6843 2.8317 2.4605 2.5872 2.3709]
Parsed loss array (first 10): [3.6027 2.9915 2.9228 3.5279 2.8761 2.8376 2.8946 2.959  2.5008 2.3316]
MeanRelativeError: 0.017991498
MeanAbsoluteError: 0.029692966
Rule,mean_relative_error 0.017991498
pass mean_relative_error=0.017991498 <= 0.05 or mean_absolute_error=0.029692966 <= 0.0002


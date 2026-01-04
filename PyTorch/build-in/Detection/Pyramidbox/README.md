# Pyramidbox

## 1. 模型概述
PyramidBox 是一种基于深度学习的人脸检测算法，特别设计用于在复杂环境下实现高效、准确的多尺度人脸检测。它通过引入特征金字塔网络（Feature Pyramid Network, FPN）结合多层次监督学习的方式，增强了模型对不同大小和分辨率人脸的检测能力，尤其是在处理小尺寸人脸时表现尤为突出。该方法能够在保证实时性的同时提升人脸检测的准确性，适用于安防监控、人脸识别系统等多种应用场景。

- 参考实现：
    ```
    url=https://github.com/yxlijun/Pyramidbox.pytorch
    commit_id=76cf3558ef09bf27df15d960f478b7e5b4a6a673
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

- 您可以点击[此链接](http://www.cs.toronto.edu/~kriz/cifar.html)下载所需的 WIDERFACE 数据集，并上传到服务器的任意路径下。

- 请你按以下目录组织解压后的 WIDERFACE 数据集：

   ```
    ├── WIDER
        ├── widerface
            ├──WIDER_train
                ├──images
                    ├──0--Parade/
                    ├──1--Handshaking/
                    ├──...
            ├──WIDER_val
                ├──images
                    ├──0--Parade/
                    ├──1--Handshaking/
                    ├──...    
            ├──wider_face_split
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Detection/Pyramidbox
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_Pyramidbox`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_Pyramidbox
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_Pyramidbox -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_Pyramidbox /bin/bash
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
    cd /workspace/Detection/Pyramidbox
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 您需要从[vgg](https://pan.baidu.com/s/1Q-YqoxJyqvln6KTcIck1tQ)链接下载vgg16_reducedfc.pth，并请放置在 './weights/' 路径下
  
  - 启动训练：
    ```
    python train.py --batech_size 4 --lr 5e-4
    ```
  - 开启混合精度amp训练：
    ```
    python amptrain.py --batech_size 4 --lr 5e-4
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: 0.000995756079854359
MeanAbsoluteError: 0.053523549488054614
Rule,mean_relative_error 0.000995756079854359
pass mean_relative_error=0.000995756079854359 <= 0.05 or mean_absolute_error=0.053523549488054614 <= 0.0002


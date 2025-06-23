# RCF

## 1. 模型概述
RCF（Richer Convolutional Features，更丰富的卷积特征）是一种用于边缘检测的技术。它利用了卷积神经网络中不同层次的特征图，这些特征图包含了从低级到高级的不同尺度的信息。通过结合这些信息，RCF能够更准确地识别出图像中的边缘。RCF模型可以进行端到端的训练，这意味着整个网络可以从原始像素值直接学习到如何生成高质量的边缘映射，而无需手工设计特征或中间处理步骤。相比传统的边缘检测算法（如Canny、Sobel等），RCF能够在保持高效的同时提供更高的检测精度，尤其是在复杂场景下表现更为出色,适用于需要高精度边缘信息的各种应用场景。

- 参考实现：
    ```
    url=https://github.com/mayorx/rcf-edge-detection
    commit_id=68341dfcadd517db8cdf502f9740b7330496cbfa
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
#### 2.2.1 获取数据集

- 请下载您需要的数据集，RCF模型可选的数据集包括 `HED-BSDS` 、 `PASCAL` 和 `NYUD` 。您可以点击
  [此链接](http://mftp.mmcheng.net/liuyun/rcf/data/bsds_pascal_train_pair.lst)
  [此链接](http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz)
  [此链接](http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz)
  [此链接](http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz)
  下载所需的数据集，并上传到服务器的任意路径下。


#### 2.2.2 解压数据集

- 解压训练数据集

以下训练均以`HED-BSDS` 和 `PASCAL` 数据集为例。

请将下载好的数据集上传到服务器的任意路径下并解压，解压后训练集图片分别位于“HED-BSDS/train/”和“PASCAL/aug_data/”文件夹路径下，该目录下每个文件夹代表一个尺度和是否有翻转，且同一文件夹下的所有图片都有相同的标签。

解压后的数据集目录结构如下所示：

```
 ├── data
    ├── HEDS-BSDS
        ├──groundTruth
        ├──test
            ├──xxx.jpg
            ├──...
        ├──train
            ├──aug_data
            ├──aug_data_scale_0.5
            ├──...
    ├── PASCAL
        ├──aug_data
        ├──aug_gt
    ├── bsds_pascal_train_pair.lst
   ```

#### 2.2.3 下载预训练模型权重

您可以点击[此链接](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)下载resnet101-5d3b4d8f预训练模型权重文件

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Detection/RCF
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_RCF`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_RCF 
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_RCF -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_RCF /bin/bash
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
    cd /workspace/Detection/RCF
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请放置数据集位于`data/`目录下。
  - 启动训练
    ```
    pthon train.py
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: 0.14154994
MeanAbsoluteError: -0.05548365
Rule,mean_absolute_error -0.05548365

Test result:
pass mean_relative_error=0.14154994 <= 0.05 or mean_absolute_error=-0.05548365 <= 0.0002

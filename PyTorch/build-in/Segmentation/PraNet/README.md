# PraNet

## 1. 模型概述
PraNet 是一种专门用于医学图像处理领域，特别是针对结肠镜图像中息肉（polyp）检测和分割的深度学习模型。它由一种创新的区域注意力机制驱动，旨在提高对息肉区域的识别精度和分割效果。PraNet 引入了一个独特的区域注意力模块，能够有效地捕捉到息肉及其周围环境之间的关系，从而增强了模型对息肉位置的敏感性。该网络利用了不同层次的特征图来获取丰富的上下文信息。通过结合低级细节特征和高级语义特征，提高了模型对息肉边界和形态的精确描绘能力。相比于传统的息肉检测方法，PraNet 在保证较高运行效率的同时，显著提升了息肉分割的准确率和召回率，这在临床诊断中尤为重要。作为一种基于深度学习的方法，PraNet 支持从原始输入图像直接学习到高质量的息肉分割结果，无需复杂的预处理或后处理步骤。


- 参考实现：
    ```
    url=https://github.com/DengPingFan/PraNet
    commit_id=74caa7bf5b4d975789acd58b96f4378a2411853f
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

- 您可以点击[此链接](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing)下载模型训练所需的训练集数据集，包括Kvasir-SEG和CVC-ClinicDB两个子数据集，并将训练集放置在 ./data/TrainDataset 路径下。
- 测试集[下载链接](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing)，请放置在 ./data/TestDataset 路径下。

- 请你按以下目录组织解压后的训练集：

   ```
    ├── data
        ├── TrainDataset
            ├──images
                ├──1.png
                ├──2.png
                ├──...
            ├──masks
                ├──1.png
                ├──2.png
                ├──...
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Segmentation/PraNet
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_PraNet`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_PraNet
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_PraNet -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_PraNet /bin/bash
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
    cd /workspace/Segmentation/PraNet
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 您需要从[此链接](https://drive.google.com/file/d/1lJv8XVStsp3oNKZHaSr42tawdMOq6FLP/view?usp=sharing)下载预训练的权重，并请放置在 'snapshots/PraNet_Res2Net/PraNet-19.pth' 路径下
  
  - 启动训练：
    ```
    python MyTrain.py
    ```
  - 开启混合精度amp训练：
    ```
    python train_amp.py
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [6.452  6.5635 6.1108 6.2119 5.7991 6.0853 6.4362 5.9842 6.0827 6.0155]
Parsed loss array (first 10): [6.6842 6.4709 6.4129 6.4765 6.1122 5.7568 6.357  6.3576 5.2967 6.3496]
MeanRelativeError: 0.022051198
MeanAbsoluteError: -0.04758262
Rule,mean_absolute_error -0.04758262
pass mean_relative_error=0.022051198 <= 0.05 or mean_absolute_error=-0.04758262 <= 0.0002


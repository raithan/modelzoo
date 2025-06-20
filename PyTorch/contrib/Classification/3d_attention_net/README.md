# 3d_attention_net

## 1. 模型概述
3d_attention_net是一种基于深度学习的图像分类模型，它利用注意力机制来增强网络性能。该网络采用了类似于 ResNet 的残差块设计，允许训练非常深的网络而不遭受梯度消失问题的影响。通过引入注意力模块，模型能够在处理信息时聚焦于重要的特征，从而提高识别准确率。这些模块可以嵌入到网络的不同层次中，以捕获不同尺度下的重要特征。借助注意力机制，网络能够有效地整合来自不同层的信息，使得最终的特征表示既包含低级细节也涵盖高级语义信息。Residual Attention Network 可被应用于多种计算机视觉任务，例如在诸如 ImageNet 等大规模数据集上进行图像级别的分类任务；用于像素级别的图像分类，比如分割出人像、道路等特定区域。

- 参考实现：
    ```
    url=https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch
    commit_id=88ed90f1b59f4b20e152495d3a5b6a19a4aa4232
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

- 请下载您需要的数据集，ResidualAttentionNetwork需要使用到CIFAR-10数据集，其由 10 个类的 60000 张 32x32 彩色图像组成，每个类有 6000 张图像。有 50000 张训练图像和 10000 张测试图像。您可以点击[此链接](http://www.cs.toronto.edu/~kriz/cifar.html)下载所需的数据集，并上传到服务器的data路径下。

- 解压后的数据集目录结构参考如下所示：

   ```
    ├── data
        ├── cifar-10-batches-py
            ├──batches.meta
            ├──readme.html
            ├──test_batch
            ├──data_batch_1
            ├──data_batch_2
            ├──data_batch_3
            ├──data_batch_4
            ├──data_batch_5
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Classification/3d_attention_net
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_3d_attention_net`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_3d_attention_net 
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_3d_attention_net -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_3d_attention_net /bin/bash
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
    cd /workspace/Classification/3d_attention_net
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请放置数据集位于`data/`目录下。
  - 请确保is_train参数为True即：is_train = True
  
  - 启动训练
    ```
    pthon train.py
    ```
  - 加入混合精度
    ```
    python train_amp.py
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: 0.09975952703366982
MeanAbsoluteError: -0.0003195767195767208

Rule,mean_absolute_error -0.0003195767195767208
pass mean_relative_error=0.09975952703366982 <= 0.05 or mean_absolute_error=-0.0003195767195767208 <= 0.0002

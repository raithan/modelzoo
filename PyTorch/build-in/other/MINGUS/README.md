# MINGUS

## 1. 模型概述
MINGUS是一个基于Transformer架构的序列到序列(Seq2Seq)神经网络模型，专门用于生成单音爵士乐旋律。它通过对和弦变化进行即兴创作来模拟音乐旋律的生成过程。该模型利用了专门设计的音高和持续时间嵌入模型，并在预测时考虑了当前及后续和弦、贝斯线以及小节内的位置等特征。因此，MINGUS能够生成与爵士乐风格相符的旋律，其表现达到了使用神经模型生成音乐的先进水平，尤其擅长于爵士音乐的生成。此外，用户还可以通过一个交互式的浏览器演示来体验由MINGUS生成的MIDI音乐文件。

- 参考实现：
    ```
    url=https://github.com/vincenzomadaghiele/MINGUS
    commit_id=3c4ac1210c6b09cc9ed8f904a0bf49336a4fd5af
    ```

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.2.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/other/MINGUS
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.2.2 执行以下命令，构建名为`sdaa_MINGUS`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_MINGUS
   ```

#### 2.2.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_MINGUS -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.2.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_MINGUS /bin/bash
    ```

#### 2.2.5 执行以下命令，启动虚拟环境。

    ```
    conda activate torch_env_py310
    ```

#### 2.2.6 执行以下命令，安装其他环境依赖包。

    ```
    pip install -r requirements.txt
    ```


### 2.3 启动训练

#### 2.3.1 在Docker环境中，进入训练脚本所在目录。
    ```
    cd /workspace/other/MINGUS
    ```

#### 2.3.2 运行以下命令处理数据集。

    ```
    python A_preprocessData/data_preprocessing.py --format xml
    ```
    成功运行将会生成DATA.json文件用于训练

#### 2.3.3 运行以下命令训练。
  
  - 启动训练：
    ```
    python3 B_train/train.py
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

(mingus) huangyun@node03:~/MINGUS-master$ python get_loss.py
Parsed loss array (first 10): [9.95 4.43 4.45 5.83 4.99 4.49 4.01 4.55 4.33 3.97]
Parsed loss array (first 10): [9.79 4.64 4.37 5.34 4.56 4.46 4.69 4.75 4.   4.62]
MeanRelativeError: -0.035348825
MeanAbsoluteError: -0.114719115
Rule,mean_absolute_error -0.114719115
pass mean_relative_error=-0.035348825 <= 0.05 or mean_absolute_error=-0.114719115 <= 0.0002


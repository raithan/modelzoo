# Emotion-recognition

## 1. 模型概述
是一个基于深度学习的实时人脸情绪识别系统。它利用卷积神经网络（CNN）分析摄像头捕捉的面部图像，能够识别出如愤怒、恐惧、快乐、悲伤、惊讶等基本情绪，并实时显示检测结果及各类情绪的概率，适用于人机交互、情感分析等场景。

- 参考实现：
    ```
    url=https://github.com/otaha178/Emotion-recognition
    commit_id=5c3b2c7bff404d244abd0798e3caaef00bf4f593
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

- 使用的开源数据集为fer2013，您可以从[此链接](https://www.kaggle.com/c/3364/download-all)下载所需数据集，请您放在fer2013/fer2013/目录下


### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Classification/Emotion-recognition
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_Emotion-recognition`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_Emotion-recognition
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_Emotion-recognition -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_Emotion-recognition /bin/bash
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
    cd /workspace/Classification/Emotion-recognition
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    python train_emotion_classifier.py
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [1.9332 1.9335 1.9271 1.957  2.0123 1.8959 1.9303 1.9157 1.9251 1.8908]
Parsed loss array (first 10): [1.9266 1.914  1.8962 1.8812 1.9466 1.9917 1.9528 1.8733 1.8417 1.879 ]
MeanRelativeError: 0.008780089
MeanAbsoluteError: 0.015480579
Rule,mean_relative_error 0.008780089
pass mean_relative_error=0.008780089 <= 0.05 or mean_absolute_error=0.015480579 <= 0.0002


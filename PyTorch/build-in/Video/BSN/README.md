# BSN

## 1. 模型概述
BSN (Boundary-Sensitive Network) 是一种用于时序动作定位的弱监督学习模型，其核心目标是从未剪辑的长视频中精确定位出动作发生的起始和结束时间。该模型分为三个阶段：首先通过 Proposal Generation Module 生成多尺度的候选片段，利用边界敏感性得分（起始与结束概率）匹配构建候选；然后使用 Proposal Evaluation Module 对每个候选片段预测其属于某一动作类别的置信度；最后通过 Localization Module 进一步优化边界位置，提升定位精度。BSN 能有效捕捉动作的边界上下文信息，在仅使用视频级别类别标签（无帧级标注）的情况下，实现高质量的动作提议生成与定位，是弱监督动作检测中的代表性方法之一。


- 参考实现：
    ```
    url=https://github.com/wzmsltw/BSN-boundary-sensitive-network/tree/master
    commit_id=f13707fbc362486e93178c39f9c4d398afe2cb2f
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

- 训练BSN模型需要使用到ActivityNet数据集，该数据集非常庞大，请你从[源码仓库中](https://github.com/wzmsltw/BSN-boundary-sensitive-network)下载提取好的特征数据集，并按以下结构阻止数据集。

   ```
    BSN
    |-- data                     
    |   |-- activitynet_annotations      
    |       |-- anet_anno_action.json
    |   |-- activitynet_feature_cuhk
    |       |-- csv_mean_100
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Video/BSN
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_BSN`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_BSN
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_BSN -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_BSN /bin/bash
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
    cd /workspace/Video/BSN
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    python main.py --module TEM --mode train
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [2.773 2.77  2.772 2.768 2.754 2.753 2.756 2.737 2.744 2.745]
Parsed loss array (first 10): [2.772 2.759 2.764 2.748 2.74  2.744 2.76  2.731 2.707 2.714]
MeanRelativeError: -0.00043977896
MeanAbsoluteError: -0.007947631
Rule,mean_absolute_error -0.007947631
pass mean_relative_error=-0.00043977896 <= 0.05 or mean_absolute_error=-0.007947631 <= 0.0002


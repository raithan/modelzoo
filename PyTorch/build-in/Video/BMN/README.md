# BMN

## 1. 模型概述
BMN (Boundary-Matching Network) 是一种用于高效生成时序动作候选片段的弱监督视频分析模型，其核心思想是将候选动作的生成过程分解为起始边界和结束边界的匹配。它首先通过一维时序网络预测每个时间点作为动作起始或结束的概率，然后将所有可能的起始-结束边界对进行外积匹配，构建一个二维的“候选匹配图”，并预测每个候选片段的置信度。这种方法避免了对大量候选片段的显式枚举与评估，大幅提升了候选生成的速度和密度，为后续的动作分类网络提供了高质量、多尺度的候选片段，广泛应用于时序动作定位任务中。


- 参考实现：
    ```
    url=https://github.com/JJBOY/BMN-Boundary-Matching-Network
    commit_id=a92c1d79c19d88b1d57b5abfae5a0be33f3002eb
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

- 训练BMN模型需要使用到ActivityNet数据集，该数据集非常庞大，请你从[源码仓库中](https://github.com/wzmsltw/BSN-boundary-sensitive-network)下载提取好的特征数据集，并按以下结构阻止数据集。

   ```
    ├── csv_mean_100
        ├── 视频1的特征csv
        ├── 视频2的特征csv
        │   ...             
        ├── 视频19228的特征csv
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Video/BMN
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_BMN`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_BMN
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_BMN -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_BMN /bin/bash
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
    cd /workspace/Video/BMN
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    python main.py --mode train
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [2.517 2.48  2.464 2.445 2.465 2.441 2.469 2.431 2.392 2.378]
Parsed loss array (first 10): [2.5   2.522 2.456 2.473 2.484 2.475 2.479 2.447 2.405 2.459]
MeanRelativeError: -0.03470299
MeanAbsoluteError: -0.07926087
Rule,mean_absolute_error -0.07926087
pass mean_relative_error=-0.03470299 <= 0.05 or mean_absolute_error=-0.07926087 <= 0.0002


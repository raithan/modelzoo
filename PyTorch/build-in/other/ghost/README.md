# ghost

## 1. 模型概述
ghost 是一个基于深度学习的人脸生成与动态驱动模型，它能够从单张静态人像照片生成可动画化的3D人脸，并支持通过语音或动作序列驱动实现面部表情和口型的自然变化，常用于虚拟人、数字人和AI合成视频等场景。该模型融合了3D人脸重建、生成对抗网络（GAN）和语音驱动口型同步技术，属于生成式AI（AIGC） 和 多模态内容生成 领域。


- 参考实现：
    ```
    url=https://github.com/ai-forever/ghost
    commit_id=44e58aad8600ee83ad4c8213aaa9698ccaf66c1c
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

- 请从[此链接](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/)下载VGGFace2数据集，将数据集上传到服务器任意路径下并解压。

- 请你按以下结构组织VGGFace2数据集：

   ```
    ├── VGGFace2
    │    ├──train
    │    │      │──n000012      
    │    │      ├──n000021
    │    │      ├──n000036
    │    │      ├──...
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/other/ghost
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_ghost`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_ghost
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_ghost -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_ghost /bin/bash
    ```

#### 2.3.5 执行以下命令，启动虚拟环境。

    ```
    conda activate torch_env_py310
    ```

#### 2.3.6 执行以下命令，安装其他环境依赖包。

    ```
    pip install -r requirements.txt
    ```

#### 2.3.6 执行以下命令，下载预训练权重等文件
    ```
    bash download_models.sh
    ```

### 2.4 启动训练

#### 2.4.1 在Docker环境中，进入训练脚本所在目录。
    ```
    cd /workspace/other/ghost
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    python train.py --run_name "test"
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [43.48785  43.347908 39.85195  40.363937 35.966484 35.39212  33.629364
 32.75888  31.58883  29.10593 ]
Parsed loss array (first 10): [47.07966  42.862076 41.583405 40.5715   35.55255  31.589716 33.632084
 30.97422  27.526873 28.931944]
MeanRelativeError: -0.0022781228
MeanAbsoluteError: -0.14936602
Rule,mean_absolute_error -0.14936602
pass mean_relative_error=-0.0022781228 <= 0.05 or mean_absolute_error=-0.14936602 <= 0.0002



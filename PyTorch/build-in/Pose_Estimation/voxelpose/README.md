# VoxelPose

## 1. 模型概述
VoxelPose 是一种基于立体视觉的实时多人 3D 姿态估计模型，它通过将多个视角的 2D 图像重投影到一个共享的 3D 体素空间中，直接在 3D 空间中进行特征学习和关键点定位，从而避免了传统方法中复杂的逐人检测和匹配过程。该模型采用卷积神经网络从 2D 特征图中提取信息，并将其变换和聚合到 3D 体素网格中，最后通过 3D 卷积操作回归出人体关节的 3D 坐标，具有良好的空间一致性和对遮挡的鲁棒性，适用于多视角相机环境下的无标记动作捕捉场景。


- 参考实现：
    ```
    url=https://github.com/microsoft/voxelpose-pytorch/
    commit_id=9ef5d407a597c9647b2c8f6c0a246b725a87a054
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

- 需要使用到Shelf和Campus数据集，您可以点击[此链接](http://campar.in.tum.de/Chair/MultiHumanPose)下载所需的数据集，并将其解压至{POSE_ROOT}/data/Shelf 和 {POSE_ROOT}/data/CampusSeq1路径下

- 请你按照以下结构组织Shelf和Campus数据集

   ```
    |-- data
        |-- Shelf
        |   |-- Camera0
        |   |-- ...
        |   |-- Camera4
        |   |-- actorsGT.mat
        |   |-- calibration_shelf.json
        |   |-- pred_shelf_maskrcnn_hrnet_coco.pkl
        |-- CampusSeq1
        |   |-- Camera0
        |   |-- Camera1
        |   |-- Camera2
        |   |-- actorsGT.mat
        |   |-- calibration_campus.json
        |   |-- pred_campus_maskrcnn_hrnet_coco.pkl
        |-- panoptic_training_pose.pkl
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Pose_Estimation/VoxelPose
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_VoxelPose`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_VoxelPose
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_VoxelPose -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_VoxelPose /bin/bash
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
    cd /workspace/Pose_Estimation/VoxelPose
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
     python run/train_3d.py --cfg configs/panoptic/resnet50/prn64_cpn80x80x20_960x512_cam5.yaml
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: 10325.806
MeanAbsoluteError: -6.524173
Rule,mean_absolute_error -6.524173
pass mean_relative_error=10325.806 <= 0.05 or mean_absolute_error=-6.524173 <= 0.0002



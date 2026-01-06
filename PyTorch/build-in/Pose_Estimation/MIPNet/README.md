# MIPNet

## 1. 模型概述
MIPNet 是一个专注于多视角三维人体姿态估计的先进模型或框架。它通过利用多个摄像头从不同角度捕捉到的图像或视频数据，来精确地重建出人体在三维空间中的姿态，包括各个关节的位置和角度。这种方法能够克服单目视觉中因遮挡或视角单一导致的信息缺失问题，从而提供更加准确和稳定的姿态估计结果。MIPNet 可以广泛应用于运动分析、虚拟现实、人机交互等领域，为实现更自然、更精确的人体动作捕捉提供支持。


- 参考实现：
    ```
    url=https://github.com/rawalkhirodkar/MIPNet.git
    commit_id=505c92ec59ac79686a217dac45eb188fc38b8499
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

- 您需要使用到coco数据集，可以通过[MSCOCO数据集官网](http://mscoco.org/)自行下载数据集，并按照以下目录组织数据集。或者用户进入到根目录，执行以下命令，下载coco数据集。coco数据集包括了图片，labels，annotations。下载完成后数据集默认存在在根目录的data文件中。

- 请你按以下结构组织coco数据集：

   ```
    ├── coco
        ├── LICENSE
        ├── README.md
        ├── annotations
            ├──instances_train2017.json
            ├──instances_val2017.json
            ├──person_keypoints_val2017.json
            ├──person_keypoints_train2017.json
        ├── images
            ├──test2017
            ├──val2017
            ├──train2017
        ├── labels
            ├──train2017
            ├──train2017.cache3
            ├──val2017
            ├──val2017.cache3 
        ├── test-dev2017.txt
        ├── train2017.txt
        ├── xxx

   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Pose_Estimation/MIPNet
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_MIPNet`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_MIPNet
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_MIPNet -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_MIPNet /bin/bash
    ```

#### 2.3.5 执行以下命令，启动虚拟环境。

    ```
    conda activate torch_env_py310
    ```

#### 2.3.6 执行以下命令，安装其他环境依赖包。

    ```
    pip install -r requirements.txt
    ```
- 注：原始项目需要使用到lshashing，若直接安装会报错，已在requirements.txt中修改了lshashing依赖为lshash3，请确保安装lshash3
      若lshash3未成功安装，请从GitHub从源码安装lshashing，存在部分拼写在python3中已被弃用，请修改后运行pip install -e .安装lshashing

#### 2.3.7 安装crowdpose。

- 执行以下命令从源码安装crowdpose。

    ```
    git clone https://github.com/Jeff-sjtu/CrowdPose.git
    cd CrowdPose/crowdpose-api/PythonAPI/
    make install
    python setup.py install --user
    ```    

- 验证是否正确安装crowdpose。

    ```
    import crowdposetools
    ```    

### 2.4 启动训练

#### 2.4.1 在Docker环境中，进入训练脚本所在目录。
    ```
    cd /workspace/Pose_Estimation/MIPNet
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 源项目中的链接已失效，您需要从[百度网盘](https://pan.baidu.com/s/1hw6EmwYdQDF_yYyc7uNolw)下载预训练的权重（mip5），并请放置在 './tools/models/pytorch/imagenet/hrnet_w48-8ef0771d.pth' 路径下
  
  - 启动训练(yaml文件可以从源项目的README.md中下载，并放在../lib/config/路径下)：
    
    ```
    python train.py --cfg ../lib/config/w48_384x288_adam_lr1e-3.yaml --gpu 0
    ```


### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: 0.085761644
MeanAbsoluteError: 7.426057e-05
Rule,mean_absolute_error 7.426057e-05
pass mean_relative_error=0.085761644 <= 0.05 or mean_absolute_error=7.426057e-05 <= 0.0002


# YOLOR

## 1. 模型概述
YOLOR是一种先进的实时目标检测算法。它基于YOLO系列模型，但引入了多项改进和优化，旨在提升速度与精度的平衡。YOLOR 是一种单阶段的目标检测器，能够在保持高检测速度的同时实现优秀的检测性能。YOLOR 的核心理念是学习一个统一的表示，该表示能够同时支持多种任务和应用场景。这种统一表示不仅包括物体的位置信息，还包含了语义信息、上下文信息等，使得模型在各种复杂的场景下都能表现良好。YOLOR 使用了经过优化的骨干网络（如CSPDarknet53），并在此基础上进行了进一步的改进，以提高特征提取的效率和效果。此外，还采用了PANet结构来增强特征融合能力。


- 参考实现：
    ```
    url=https://github.com/WongKinYiu/yolor
    commit_id=3ca250ae2247ca13911fa498cbe8e2c9b6bab5b0
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
        ├── train2017.cache
        ├── val2017.cache
        ├── val2017.txt

   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Detection/YOLOR
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_YOLOR`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_YOLOR
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_YOLOR -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_YOLOR /bin/bash
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
    cd /workspace/Detection/YOLOR
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 您需要从[此链接](https://drive.google.com/file/d/1lJv8XVStsp3oNKZHaSr42tawdMOq6FLP/view?usp=sharing)下载预训练的权重，并请放置在 'snapshots/PraNet_Res2Net/PraNet-19.pth' 路径下
  
  - 启动训练：
    ```
    python train.py --batch-size 8 --img 1280 1280 --data coco.yaml --cfg cfg/yolor_p6.cfg --weights '' --name yolor_p6 --hyp hyp.scratch.1280.yaml
    ```
  - 开启混合精度amp训练：
    ```
    python train_amp.py --batch-size 4 --img 1280 1280 --data coco.yaml --cfg cfg/yolor_p6.cfg --weights '' --name yolor_p6 --hyp hyp.scratch.1280.yaml
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: 0.0026242519
MeanAbsoluteError: 0.00045768163
Rule,mean_absolute_error 0.00045768163
pass mean_relative_error=0.0026242519 <= 0.05 or mean_absolute_error=0.00045768163 <= 0.0002


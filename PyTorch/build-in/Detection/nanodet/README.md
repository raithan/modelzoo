# nanodet

## 1. 模型概述
NanoDet 是一款专为移动端和边缘设备设计的轻量级、高性能目标检测模型，采用无锚框（anchor-free）架构，具有模型小、速度快、部署简单的特点，能够在保持较高检测精度的同时，在手机或嵌入式设备上实现实时物体识别，广泛应用于智能监控、移动视觉和工业检测等场景。

- 参考实现：
    ```
    url=https://github.com/RangiLyu/nanodet
    commit_id=be9b4a9001d7f9b6fc89c2df31ae8d428e35b4f0
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
    cd <modelzoo-dir>/PyTorch/Detection/nanodet
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_nanodet`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_nanodet
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_nanodet -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_nanodet /bin/bash
    ```

#### 2.3.5 执行以下命令，启动虚拟环境。

    ```
    conda activate torch_env_py310
    ```

#### 2.3.6 执行以下命令，安装其他环境依赖包。

    ```
    pip install -r requirements.txt
    python setup.py develop
    ```


### 2.4 启动训练

#### 2.4.1 在Docker环境中，进入训练脚本所在目录。
    ```
    cd /workspace/Detection/nanodet
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集，并修改config/nanodet-plus-m_320.yml文件中的数据集路径。
  
  - 启动训练：
    ```
    python tools/train.py config/nanodet-plus-m_320.yml
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [4.5952 4.5609 4.5309 4.4733 4.4101 4.4412 4.4357 4.4178 4.3402 4.4041]
Parsed loss array (first 10): [4.6026 4.5747 4.507  4.4876 4.4596 4.4588 4.4453 4.4442 4.4358 4.4035]
MeanRelativeError: -0.001252093
MeanAbsoluteError: -0.0064336243
Rule,mean_absolute_error -0.0064336243
pass mean_relative_error=-0.001252093 <= 0.05 or mean_absolute_error=-0.0064336243 <= 0.0002



# YOLOv6

## 1. 模型概述

YOLOv6是yolo系列的经典算法，在计算机视觉领域具有广泛的应用前景。
链接：https://github.com/ultralytics/ultralytics/tree/main

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 数据集准备
#### 2.2.1 获取数据集


- 根据如下链接下载

[coco2017labels.zip](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip)

[train2017.zip](http://images.cocodataset.org/zips/train2017.zip)

[val2017.zip](http://images.cocodataset.org/zips/val2017.zip)

- 执行如下命令解压

```
mkdir datasets
unzip -q coco2017labels.zip -d datasets
unzip -q train2017.zip -d datasets/coco/images
unzip -q val2017.zip -d datasets/coco/images
```


#### 2.2.2 数据集目录结构参考如下所示:

```
├── datasets #根目录
  ├── coco 
      ├── annotations #json标注目录
      │   └── instances_val2017.json #对应目标检测、分割任务的验证集标注文件
      ├── images
      │   ├── train2017 #训练集图片，约118287张
      │   └── val2017 #验证集图片，约5000张
      ├── labels  #txt标注目录
      │   ├── train2017 #对应目标检测的训练集txt标注文件
      │   └── val2017 #对应目标检测的验证集txt标注文件
      ├── train2017.txt #训练集图片路径
      └── val2017.txt #验证集图片路径

```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境（示例）。

1. 执行以下命令，进入Dockerfile所在目录。
    ```
    cd <modelzoo-dir>/PyTorch/Detection/YOLO11
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

2. 执行以下命令，构建名为`sdaa_yolov6`的镜像。
    ```
    DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_yolo11
    ```
3. 执行以下命令，启动容器。
    ```
    docker run -itd --name yolo11pt -v <dataset_path>:/workspace/Detection/YOLO11/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_yolo11 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。

4. 执行以下命令，进入容器。
    ```
    docker exec -it yolo11pt /bin/bash
    ```

5. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env_py310
    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Detection/YOLOv6
    ```
    

2. 训练指令。
    
        python run_scrips/run_demo.py \
            --model ultralytics/cfg/models/v6/yolov6.yaml \ #模型配置文件
            --data ultralytics/cfg/datasets/coco128.yaml \  #数据集配置文件，此处使用官方测试数据集coco128测试
            --epochs 25 \
            --batch 32 \
            --device [0,1] \ # [-1,0,1,2]就是开启分布式训练，四张卡
            --optimizer SGD \
            --close_mosaic 0 \
            --workers 0 \
            --project runs/train \
            --name exp
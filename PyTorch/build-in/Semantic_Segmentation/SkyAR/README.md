# SkyAR

## 1. 模型概述
SkyAR 是一个基于深度学习的图像天空替换与美化模型，它通过语义分割技术精确识别图像中的天空区域，然后将原天空无缝替换为新的天空图像，并结合边缘优化和光照融合算法，实现自然逼真的视觉效果。该模型支持静态图像和视频处理，能够自动调整颜色、亮度和阴影匹配，使合成结果更加真实，广泛应用于图像编辑、增强现实和视觉创意等领域。


- 参考实现：
    ```
    url=https://github.com/jiupinjia/SkyAR
    commit_id=1d5fe73409188b834d8bf5bb844ef01a60e3265a
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

- 请你下载原始仓库，将SkyAR/datases.zip上传到服务器任意路径下并解压。


### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/cv-semantic_segmentation/SkyAR
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_SkyAR`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_SkyAR
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_SkyAR -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_SkyAR /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_SkyAR /bin/bash
    ```

#### 2.3.5 执行以下命令，启动虚拟环境。

    ```
    conda activate torch_env_py310
    ```

#### 2.3.6 执行以下命令，安装其他环境依赖包。

    ```
    pip install -r requirements.txt
    ```

#### 2.3.7 下载预训练权重

    从[谷歌云盘](https://drive.usercontent.google.com/download?id=1COMROzwR4R_7mym6DL9LXhHQlJmJaV0J&export=download)下载预训练权重，上传至服务器并解压

### 2.4 启动训练

#### 2.4.1 在Docker环境中，进入训练脚本所在目录。
    ```
    cd /workspace/cv-semantic_segmentation/SkyAR
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    python train.py        --dataset cvprw2020-ade20K-defg         --checkpoint_dir checkpoints_G_coord_resnet50   --vis_dir val_out         --in_size 384   --max_num_epochs 200    --lr 1e-4       --batch_size 8  --net_G coord_resnet50
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [2.6082620e-01 1.9135552e-02 2.5896378e-02 3.9740501e-04 2.9322410e-03
 3.1464234e-02 2.7895770e-03 2.6403602e-02 2.2923001e-05 3.1263221e-02]
Parsed loss array (first 10): [4.6991578e-01 4.9282517e-02 4.0071532e-03 3.1004738e-02 5.4180401e-04
 3.0598078e-02 2.0293600e-04 2.9733872e-02 2.9565528e-02 5.5123000e-05]
MeanRelativeError: 1462.7891
MeanAbsoluteError: -0.0022559827
Rule,mean_absolute_error -0.0022559827
pass mean_relative_error=1462.7891 <= 0.05 or mean_absolute_error=-0.0022559827 <= 0.0002


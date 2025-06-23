# Retinaface

## 1. 模型概述
RetinaFace 是一种先进的人脸检测模型，它能够在各种环境下精确地定位图像中的人脸位置。与之前的人脸检测方法相比，RetinaFace 强调了对多尺度人脸的高效检测以及更高的准确性，即使是在具有挑战性的条件下（如遮挡、侧脸和低分辨率等）。该模型采用了深度卷积神经网络，并结合了特征金字塔网络来增强对不同尺度人脸的检测能力。此外，RetinaFace 还引入了额外的监督信号，通过在训练过程中增加对人脸关键点的预测来进一步提升检测精度。因此，RetinaFace 在多种应用场景中表现出色，包括安防监控、人脸识别系统及智能相册管理等。

- 参考实现：
    ```
    url=https://github.com/biubug6/Pytorch_Retinaface
    commit_id=b984b4b775b2c4dced95c1eadd195a5c7d32a60b
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

- 您可以点击[此链接](http://www.cs.toronto.edu/~kriz/cifar.html)下载所需的 WIDERFACE 数据集，并上传到服务器的任意路径下。

- 请您从[百度网盘](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA)或[Dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0)下载注释（人脸边界框）

- 解压后的数据集请确保如以下目录结构所示：

   ```
    ├── data
        ├── widerface
            ├──train
                ├──images
                    ├──0--Parade/
                    ├──1--Handshaking/
                    ├──...
                ├──label.txt
            ├──val
                ├──images
                    ├──0--Parade/
                    ├──1--Handshaking/
                    ├──...
                ├──wider_val.txt       
   ```
wider_val.txt仅包含了val的文件名，不包括标签信息。

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Detection/Retinaface
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_Retinaface`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_Retinaface
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_Retinaface -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_Retinaface /bin/bash
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
    cd /workspace/Detection/Retinaface
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请放置数据集位于`data/`目录下。
  - 预训练模型可以从[百度网盘](https://pan.baidu.com/s/12h97Fy1RYuqMMIV-RpzdPg)或[谷歌云](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1)下载，并请放置在 './weights/' 路径下
  
  - 将 resnet50 作为训练模型的骨干网络训练 Retinaface模型：
    ```
    SDAA_VISIBLE_DEVICES=0,1,2,3 python train.py --network resnet50 or
    ```
  - 将 mobile0.25 作为训练模型的骨干网络训练 Retinaface模型：
    ```
    SDAA_VISIBLE_DEVICES=0 python train.py --network mobile0.25
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: 0.026740791
MeanAbsoluteError: 0.8330229
Rule,mean_relative_error 0.026740791
pass mean_relative_error=0.026740791 <= 0.05 or mean_absolute_error=0.8330229 <= 0.0002

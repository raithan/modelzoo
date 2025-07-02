# FaceBoxes

## 1. 模型概述
FaceBoxes是一种专为实时人脸检测设计的轻量级卷积神经网络模型。FaceBoxes采用了多个不同尺度的卷积层来处理输入图像，从而确保对不同大小的人脸都有良好的检测效果。其特点是通过使用“快速下采样”策略来增加感受野的同时保持计算成本的可控。在模型的前端，FaceBoxes设计了特别的RDCL，这些层能够以较少的计算代价迅速处理输入图像，加快了整个网络的前向传播速度，有利于实现实时检测。为了提高对不同尺寸人脸的检测能力，FaceBoxes在网络中集成了多尺度的卷积层。这使得模型可以有效地捕捉到从非常小到较大的人脸区域。FaceBoxes在网络结构中利用了CReLU（Concatenated ReLU），这是一种将ReLU函数输出与其相反数拼接起来作为新特征的方法，有助于减少参数数量并加速训练过程。


- 参考实现：
    ```
    url=https://github.com/zisianw/FaceBoxes.PyTorch
    commit_id=9bc5811fe8c409a50c9f23c6a770674d609a2c3a
    ```


## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 编译nms

    ```
    ./make.sh
    ```

### 2.2 准备数据集

- 您可以点击[WIDER_FACE](https://aistudio.baidu.com/aistudio/datasetdetail/4336)下载模型训练所需的WIDER_FACE数据集，并将其放置在 $FaceBoxes_ROOT/data/WIDER_FACE 路径下。

- 请执行以下命令组织数据集结构

    ```
    #$FaceBoxes_ROOT 为项目根目录
    cd $FaceBoxes_ROOT/data
    mkdir WIDER_FACE
    cd WIDER_FACE
    #传WIDER_train.zip至WIDER_FACE
    #下载文件格式转换脚本
    git clone https://github.com/akofman/wider-face-pascal-voc-annotations.git
    mv WIDER_train.zip wider-face-pascal-voc-annotations
    cd wider-face-pascal-voc-annotations
    unzip WIDER_train.zip
    ./convert.py -ap ./wider_face_split/wider_face_train_bbx_gt.txt -tp ./WIDER_train_annotations/ -ip ./WIDER_train/images/
    mv WIDER_train_annotations annotations
    mv annotations $FaceBoxes_ROOT/data/WIDER_FACE
    mv WIDER_train/images $FaceBoxes_ROOT/data/WIDER_FACE
    cd $FaceBoxes_ROOT/data/WIDER_FACE
    cp gen_train_data.py $FaceBoxes_ROOT/data/WIDER_FACE
    cd $FaceBoxes_ROOT/data/WIDER_FACE
    python gen_train_data.py
    ```

- 组织解压后的训练集目录如下所示：

   ```
    ├── data
        ├── WIDER_FACE
            ├──images
                ├──0--Parade
                ├──1--Handshaking
                ├──...
            ├──annotations
                ├──9_Press_Conference_Press_Conference_9_97.xml
                ├──9_Press_Conference_Press_Conference_9_946.xml
                ├──...
            ├──img_list.txt
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Detection/FaceBoxes
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_FaceBoxes`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_FaceBoxes
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_FaceBoxes -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_FaceBoxes /bin/bash
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
    cd /workspace/Detection/FaceBoxes
    ```

#### 2.4.2 运行以下命令训练。

  - 启动训练：
    ```
    python train.py
    ```
  - 开启混合精度amp训练：
    ```
    python amp_train.py
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: 0.022172645
MeanAbsoluteError: 0.024680909
Rule,mean_relative_error 0.022172645
pass mean_relative_error=0.022172645 <= 0.05 or mean_absolute_error=0.024680909 <= 0.0002


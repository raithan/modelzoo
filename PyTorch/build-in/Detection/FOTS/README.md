# YOLOR

## 1. 模型概述
FOTS主要用于自然场景图像中的文字定位，能够快速、准确地检测出图片中带有方向（如倾斜、旋转）的文字区域。该项目专注于检测部分，不包含识别，其训练过程结合了合成数据（SynthText）和真实场景数据（ICDAR15），在标准测试集上取得了 83.3% 的 H-mean 检测精度，在自然场景图像中文字定位的应用场景具有良好的效果。


- 参考实现：
    ```
    url=https://github.com/Wovchena/text-detection-fots.pytorch
    commit_id=04b969c87491630dce38cdb48932aa33115798c6
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

- 可选用的开源数据集包括SynthText、ICDAR2015等，其中SynthText用于pretrain，ICDAR2015用于finetune和test，将数据集上传到服务器任意路径下并解压。

- 以SynthText数据集进行训练为例，请你按以下结构组织SynthText数据集：

   ```
    ├── SynthText 
    │    ├──类别1├──序号1──图片1、2、3、4 
    │    │      │  ...         
    │    │      ├──序号xx──图片1、2、3、4 
    │    │                      
    │    ├──类别2├──序号1──图片1、2、3、4 
    |           |  ...
    |           ├──序号xx──图片1、2、3、4
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Detection/FOTS
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_FOTS`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_FOTS
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_FOTS -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_FOTS /bin/bash
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
    cd /workspace/Detection/FOTS
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
     python3 train.py --train-folder /data/xxx/SynthText --batch-size 4 --batches-before-train 2
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: 0.041522343
MeanAbsoluteError: -0.107711375
Rule,mean_absolute_error -0.107711375
pass mean_relative_error=0.041522343 <= 0.05 or mean_absolute_error=-0.107711375 <= 0.0002


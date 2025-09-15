# MobileNetV3

## 1. 模型概述
MobileNetV3 是一种轻量级卷积神经网络，专为移动和嵌入式设备设计，在保持高精度的同时显著降低计算量和模型大小。它通过结合神经架构搜索（NAS） 和改进的网络结构（如引入 squeeze-and-excite 模块、非对称卷积等），在图像分类、目标检测、语义分割等计算机视觉任务中实现了速度与精度的最佳平衡，广泛应用于手机、边缘设备等资源受限场景。

- 参考实现：
    ```
    url=https://github.com/xiaolai-sqlai/mobilenetv3
    commit_id=1a7088358ec7f27d8917d5c32c9d1182ab79a711
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

- 需要使用LSVRC2012数据集，将数据集上传到服务器任意路径下并解压。

- 请你按以下结构组织LSVRC2012数据集：

   ```
    ├── LSVRC2012
    │    ├──train
    │    │      │──n01440764     
    │    │      ├──n01443537
    │    │      ├──...    
    │    ├──val
    │    │      │──n01440764     
    │    │      ├──n01443537
    │    │      ├──...
    │    ├──train_list.txt
    │    ├──val.txt
   ```
说明：n01440764为类别ID其目录下存放着对应的图片，train_list.txt为训练数据列表，val.txt为验证数据列表。

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Classification/MobileNetV3
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_MobileNetV3`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_MobileNetV3
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_MobileNetV3 -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g  /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_MobileNetV3 /bin/bash
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
    cd /workspace/Classification/MobileNetV3
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    python main.py --model mobilenet_v3_small --epochs 300 --batch_size 256 --lr 4e-3 --update_freq 2 --model_ema false --model_ema_eval false --use_amp true --data_path path/ILSVRC2012/ --output_dir ./checkpoint
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [5.0823 5.0823 5.0817 4.9084 4.9554 4.7327 4.7572 4.5961 4.5723 4.4004]
Parsed loss array (first 10): [5.13   5.13   5.1346 4.949  4.9502 4.7942 4.7697 4.6038 4.6302 4.4792]
MeanRelativeError: -0.028756876
MeanAbsoluteError: -0.08091575
Rule,mean_absolute_error -0.08091575
pass mean_relative_error=-0.028756876 <= 0.05 or mean_absolute_error=-0.08091575 <= 0.0002



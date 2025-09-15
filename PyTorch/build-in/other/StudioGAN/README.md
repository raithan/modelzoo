# StudioGAN

## 1. 模型概述
PyTorch-StudioGAN 是一个基于 PyTorch 框架开发的高质量图像生成模型库，专注于生成对抗网络（GANs）。它为研究人员和开发者提供了一个灵活且高效的平台，用于设计、训练以及评估各种GAN模型。通过PyTorch-StudioGAN，用户可以探索不同的GAN架构和技术，如BigGAN、StyleGAN等，并能够对生成图像的质量、多样性等方面进行深入分析与改进。该库支持多种先进的训练技巧和损失函数，旨在促进GAN技术的发展及其在图像生成领域的应用。


- 参考实现：
    ```
    url=https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
    commit_id=947b35e9835b67860fdce44d337f6d7fee7c8db3
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

- 可选用的开源数据集包括CIFAR10/CIFAR100/Tiny ImageNet等，本次适配选择CIFAR10数据集，将数据集上传到服务器任意路径下并解压。

- 请按以下结构组织CIFAR10数据集：

   ```
    ├── cifar-10
    │    ├──data_batch_1
    │    ├──data_batch_2
    │    ├──...
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/other/StudioGAN
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_StudioGAN`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_StudioGAN
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_StudioGAN -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_StudioGAN /bin/bash
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
    cd /workspace/other/StudioGAN
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    python3 src/main.py -t -metrics is fid prdc -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH
    ```
CONFIG_PATH：请在src/configs/路径下选择对应的congif配置文件。
DATA_PATH：请输入数据集路径。
SAVE_PATH：请输入保存训练结果路径。

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [2.3138 2.2647 2.2257 2.1901 2.2788 2.1265 2.1847 2.269  2.1255 2.1646]
Parsed loss array (first 10): [2.3505 2.3223 2.2693 2.2205 2.1606 2.2221 2.1787 2.1467 2.2162 2.1278]
MeanRelativeError: 0.030168235
MeanAbsoluteError: 0.059064757
Rule,mean_relative_error 0.030168235
pass mean_relative_error=0.030168235 <= 0.05 or mean_absolute_error=0.059064757 <= 0.0002


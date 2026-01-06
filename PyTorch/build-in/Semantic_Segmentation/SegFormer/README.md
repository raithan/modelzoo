# SegFormer

## 1. 模型概述
SegFormer 是由 NVIDIA 研究院于 2021 年提出的一种高效、轻量级的语义分割模型，属于基于 Transformer 的图像分割方法。它采用分层的 Mix Transformer（MiT）作为编码器，提取多尺度特征，并结合一个轻量级的 MLP 解码器进行融合，无需位置编码即可实现强大的上下文建模能力。该模型在保持高精度的同时具有较快的推理速度，在 ADE20K、Cityscapes 等主流分割数据集上表现优异，适用于从移动端到高性能场景的广泛需求，是目前语义分割领域的重要基准模型之一。


- 参考实现：
    ```
    url=https://github.com/NVlabs/SegFormer
    commit_id=65fa8cfa9b52b6ee7e8897a98705abf8570f9e32
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

- 训练SegFormer使用ADE20K数据集，将数据集上传到服务器任意路径下并解压。
- 从[onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/Ept_oetyUGFCsZTKiL_90kUBy5jmPV65O5rJInsnRCDWJQ?e=CvGohw)下载权重放置在./pretrained/目录下


### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Semantic_Segmentation/SegFormer
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_SegFormer`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_SegFormer
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_SegFormer -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_SegFormer /bin/bash
    ```

#### 2.3.5 执行以下命令，启动虚拟环境。

    ```
    conda activate torch_env_py310
    ```

#### 2.3.6 执行以下命令，安装其他环境依赖包。

    ```
    pip install -r requirements.txt
    pip install -e .
    ```


### 2.4 启动训练

#### 2.4.1 在Docker环境中，进入训练脚本所在目录。
    ```
    cd /workspace/Semantic_Segmentation/SegFormer
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    python tools/train.py local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py 
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))
Parsed loss array (first 10): [2.4527457 3.3676627 3.4881365 4.1632676 3.8413515 4.3615646 4.85275
 4.2584143 4.896121  3.7872593]
Parsed loss array (first 10): [4.4952 4.8359 4.5663 4.6536 4.108  5.3919 5.1073 5.2385 4.8815 5.2035]
MeanRelativeError: -0.0025219223
MeanAbsoluteError: -0.26013926
Rule,mean_absolute_error -0.26013926
pass mean_relative_error=-0.0025219223 <= 0.05 or mean_absolute_error=-0.26013926 <= 0.0002
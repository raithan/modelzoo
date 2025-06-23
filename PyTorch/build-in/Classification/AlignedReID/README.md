# AlignedReID

## 1. 模型概述
AlignedReID提取了图像中的全局特征并与局部特征联合学习，局部特征学习通过计算两组局部特征之间的最短路径来执行对齐/匹配而无需额外监督，在联合学习后只保留全局特征来计算图像之间的相似度。AlignedReID是第一个在market1501数据集上超越人类水平的ReID方法。

- 参考实现：
    ```
    url=https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch
    commit_id=2e2d45450d69a3a81e15d18fe85c2eebbde742e4 
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
#### 2.2.1 获取数据集

AlignedReID 运行在 `market1501` 数据集上，这是一个在行人重识别领域广泛使用的数据集。它包含了来自13个不同场景的1501个行人图像，每个行人有至少一张图像，部分行人有多达五张图像。您可以点击[此链接](https://gitcode.com/Universal-Tool/6378f/?utm_source=article_gitcode_universal&index=bottom&type=card&)从公开网站中下载数据集。


#### 2.2.2 解压数据集

- 解压训练数据集

请将下载好的market1501数据集上传到服务器的任意路径下并解压。

解压后的数据集目录结构如下所示：:
```
   ├── market1501
         ├──images
              ├──xxx.jpg
              ├──...
              ├──...
         ├──ori_to_new_im_name.pkl
         ├──partitions.pkl
   ```

#### 2.2.3 配置数据集路径

该项目要求您配置数据集路径。在 中，根据您在准备数据集时使用的保存路径修改以下代码段（aligned_reid/dataset/__init__.py）。

```
# In file aligned_reid/dataset/__init__.py

########################################
# Specify Directory and Partition File #
########################################

if name == 'market1501':
  im_dir = ospeu('~/Dataset/market1501/images')
  partition_file = ospeu('~/Dataset/market1501/partitions.pkl')

elif name == 'cuhk03':
  im_type = ['detected', 'labeled'][0]
  im_dir = ospeu(ospj('~/Dataset/cuhk03', im_type, 'images'))
  partition_file = ospeu(ospj('~/Dataset/cuhk03', im_type, 'partitions.pkl'))

elif name == 'duke':
  im_dir = ospeu('~/Dataset/duke/images')
  partition_file = ospeu('~/Dataset/duke/partitions.pkl')

elif name == 'combined':
  assert part in ['trainval'], \
    "Only trainval part of the combined dataset is available now."
  im_dir = ospeu('~/Dataset/market1501_cuhk03_duke/trainval_images')
  partition_file = ospeu('~/Dataset/market1501_cuhk03_duke/partitions.pkl')

```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。
    ```
    cd <modelzoo-dir>/PyTorch/Classification/AlignedReID
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_AlignedReID`的镜像。
    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_AlignedReID
   ```

#### 2.3.3 执行以下命令，启动容器。
    ```
    docker run  -itd --name sdaa_AlignedReID -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。
    ```
    docker exec -it sdaa_AlignedReID /bin/bash
    ```

#### 2.3.5 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env_py310
    ```


### 2.4 启动训练
#### 2.4.1 在Docker环境中，进入训练脚本所在目录。
    ```
    cd /workspace/Classification/AlignedReID
    ```

#### 2.4.2 运行训练。在以下命令中指定用于保存测试日志的 experiment 目录和下载的路径。


    - 单机单卡
    ```
    python script/experiment/train.py \
    -d '(0,)' \
    --dataset market1501 \
    --normalize_feature false \
    -glw 1 \
    -llw 0 \
    -idlw 0 \
    --only_test true \
    --exp_dir SPECIFY_AN_EXPERIMENT_DIRECTORY_HERE \
    --model_weight_file THE_DOWNLOADED_MODEL_WEIGHT_FILE
    ```

    - 设置单机两卡
    ```
    python script/experiment/train.py \
    -d '((0,), (1,))' \
    --dataset market1501 \
    --normalize_feature false \
    -glw 1 \
    -llw 0 \
    -idlw 0 \
    --only_test true \
    --exp_dir SPECIFY_AN_EXPERIMENT_DIRECTORY_HERE \
    --model_weight_file THE_DOWNLOADED_MODEL_WEIGHT_FILE
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: 3.607480755653065
MeanAbsoluteError: 0.016081666666666664
Rule,mean_absolute_error 0.016081666666666664
fail mean_relative_error=3.607480755653065 <= 0.05 or mean_absolute_error=0.016081666666666664 <= 0.0002



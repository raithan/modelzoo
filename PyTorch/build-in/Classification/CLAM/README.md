# CLAM

## 1. 模型概述
CLAM是一种用于全切片图像（Whole Slide Images, WSI）的弱监督病理学分析框架。它利用多实例学习（MIL）和注意力机制，能够在仅拥有图像级别标签（如癌症/非癌症）的情况下，对大规模病理切片进行分类、定位和生存预测，同时生成可解释的热图，帮助病理学家识别关键的组织区域，在癌症诊断和预后评估中具有重要应用价值


- 参考实现：
    ```
    url=https://github.com/mahmoodlab/CLAM
    commit_id=53e2409d4a8189c682c173382964a85f114f923c
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

- 可选用的开源数据集包括camelyon16、camelyon17、TCGA等，以下实验采用camelyon17数据集，请下载后将数据集上传到服务器任意路径下并解压。

- 图片分割与打补丁：使用create_patches_fp.py 脚本处理下载的数据集
    ```
    python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch
    ```
其中，DATA_DIRECTORY 是存放原始 WSI 文件的文件夹，RESULTS_DIRECTORY 是输出结果的文件夹，会生成 masks（分割掩码）、patches（补丁坐标 .h5 文件）等。

- 特征提取：使用预训练的模型（如 ResNet50, UNI, CONCH）为每个图像块提取特征向量，而不是保存原始图像块，以节省存储空间。
    ```
    python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs
    ```
    其中，DIR_TO_COORDS 是存放补丁坐标 .h5 文件的文件夹，DATA_DIRECTORY 是存放原始 WSI 文件的文件夹，CSV_FILE_NAME 是存放补丁坐标 .csv 文件的文件夹，FEATURES_DIRECTORY 是存放特征向量的文件夹。

- 准备训练数据集：将提取的特征文件组织到 DATA_ROOT_DIR 下的子文件夹中，然后创建一个 CSV 文件，包含 case_id（患者ID）、slide_id（切片ID）和 label（标签）等列。

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Classification/CLAM
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_CLAM`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_CLAM
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_CLAM -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_CLAM /bin/bash
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
    cd /workspace/Classification/CLAM
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2处理数据集。
  - 启动训练：
    ```
    CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 0.75 --exp_code task_1_tumor_vs_normal_CLAM_75 --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --log_data --data_root_dir Extracted_feature
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [0.7134 0.7016 0.7391 0.6864 0.6952 0.6988 0.6864 0.6856 0.6842 0.6642]
Parsed loss array (first 10): [0.7134 0.7015 0.7395 0.6864 0.6956 0.698  0.6868 0.6857 0.6823 0.6615]
MeanRelativeError: 0.00085525925
MeanAbsoluteError: 0.0003910746
Rule,mean_absolute_error 0.0003910746
pass mean_relative_error=0.00085525925 <= 0.05 or mean_absolute_error=0.0003910746 <= 0.0002


# PCB

## 1. 模型概述
这实现了在 Market-1501 数据集上对 PCB 的训练，主要基于 [syfafterzy/PCB](https://github.com/syfafterzy/PCB_RPP_for_reID)仓库进行修改,仅支持100step运行。
- 仓库链接：[syfafterzy/PCB](https://github.com/syfafterzy/PCB_RPP_for_reID)

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
Market-1501 [BaiduYun](https://pan.baidu.com/s/1ntIi2Op?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=)

#### 2.2.2 处理数据集
略

### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip install -r requirements.txt
    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/build-in/Classification/PCB
    ```

2. 运行训练。该模型支持单机单卡。

    ```
    export TORCH_SDAA_AUTOLOAD=cuda_migrate  #自动迁移环境变量
    torchrun --nproc_per_node=4 PCB.py     --distributed     -d market     -a resnet50     -b 64     -j 8     --epochs 1     --logs-dir logs/market-1501/PCB/     --combine-trainval     --features 256     --height 384     --width 128     --step-size 40     --data-dir Market-1501-v15.09.15 --max-steps 100
    ```

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./loss.py)）: 





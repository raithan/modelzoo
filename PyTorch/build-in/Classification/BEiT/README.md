
# BEIT
## 1. 模型概述
BEiT是一种自监督视觉表示模型，提出了一种用于预训练视觉Transformer的masked image modeling任务，主要目标是基于损坏的图像patch块恢复原始视觉token。

- 参考实现：  
  url=https://github.com/microsoft/unilm.git
  commit_id=006195f51b10ac44773cb62bad854fdfebb3c6c8

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
BEiT 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

#### 2.2.2 处理数据集
具体配置方式可参考：https://blog.csdn.net/xzxg001/article/details/142465729。


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
  cd <ModelZoo_path>/PyTorch/contrib/Classification/BEiT
  ```

2. 运行训练。该模型支持单机单卡。
  ```
  #BEiT fine-tuning
  export TORCH_SDAA_AUTOLOAD=cuda_migrate #自动迁移环境变量
  python -m torch.distributed.launch  --nproc_per_node=4  run_class_finetuning.py --model beit_base_patch16_224 --data_path <imagenet_path> --maxstep 100 --batch_size 64 --lr 5e-4
  ```
  
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./loss.py)）
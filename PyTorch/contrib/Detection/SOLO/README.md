# SOLO
## 1. 模型概述
SOLO（Segmenting Objects by LOcations）是一种无需依赖检测框的实例分割方法，其核心思想是将实例分割任务转化为“位置-类别-掩码”三元组的预测。

- 论文链接：https://arxiv.org/abs/1912.04488
- 仓库链接：https://github.com/open-mmlab/mmdetection/tree/main/configs/solo

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
<MODLE SOLO>使用 COCO2017 数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/#download) 下载。

#### 2.2.2 处理数据集
具体配置方式可参考：https://github.com/Atten4Vis/ConditionalDETR/blob/main/README.md。


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip3 install  -U openmim 
    pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
    pip3 install opencv_python mmcv --no-deps
    mim install -e .
    pip install -r requirements.txt
    ```
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
  ```
  cd <ModelZoo_path>/PyTorch/contrib/Detection/SOLO/run_scripts
  ```
2. 运行训练。该模型支持单机单卡。
  ```
  python run_SOLO.py --config ../configs/solo/solo_r50_fpn_1x_coco.py --launcher pytorch --nproc-per-node 1 --amp --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/coco/" "val_dataloader.dataset.data_root=/data/teco-data/coco/" 
  ```
    更多训练参数参考 run_scripts/argument.py
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: 0.012749534886509534
MeanAbsoluteError: 0.04467878580093384
Rule,mean_relative_error 0.012749534886509534
pass mean_relative_error=0.012749534886509534 <= 0.05 or mean_absolute_error=0.04467878580093384 <= 0.0002
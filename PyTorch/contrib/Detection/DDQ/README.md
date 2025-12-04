# DDQ
## 1. 模型概述
DDQ模型（Dynamic Dense Query）是一种端到端密集目标检测器，核心创新是为每个空间位置引入可学习的动态查询机制，
通过动态融合和自适应调整，极大提升了检测精度与效率。

- 论文链接：https://arxiv.org/abs/2303.12776
- 仓库链接：https://github.com/open-mmlab/mmdetection/tree/main/configs/ddq

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
<MODLE DDQ>使用 COCO2017 数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/#download) 下载。

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
    pip3 install  -U openmim 
    pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
    pip3 install opencv_python mmcv --no-deps
    mim install -e .
    pip install -r requirements.txt
    ```
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
  ```
  cd <ModelZoo_path>/PyTorch/contrib/Detection/DDQ/run_scripts
  ```
2. 运行训练。该模型支持单机单卡。
  ```
  python run_DDQ.py --config ..configs/ddq/ddq-detr-4scale_r50_8xb2-12e_coco.py --launcher pytorch --nproc-per-node 1 --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/coco/" "val_dataloader.dataset.data_root=/data/teco-data/coco/"
  ```
# 官方 PyTorch 明确禁止在 autocast 场景下用 BCELoss 或 binary_cross_entropy，因为这两者在半精度（float16）下数值不安全，执行AMP会报错。
    更多训练参数参考 run_scripts/argument.py
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: 0.30401079610065357
MeanAbsoluteError: -5.358135464191437
Rule,mean_absolute_error -5.358135464191437
pass mean_relative_error=0.30401079610065357 <= 0.05 or mean_absolute_error=-5.358135464191437 <= 0.0002
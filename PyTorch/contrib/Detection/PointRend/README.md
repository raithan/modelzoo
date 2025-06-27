# PointRend
## 1. 模型概述
传统实例分割方法如 Mask R-CNN 对掩码进行像素级分类，但常采用固定分辨率的掩码输出，难以精细捕捉目标边界细节。
PointRend（Point-based Rendering）提出了一种高效且精细的掩码预测策略，通过选取重要点进行逐步细化，实现更高质量的分割结果。

- 论文链接：https://arxiv.org/abs/1912.08193
- 仓库链接：https://github.com/open-mmlab/mmdetection/tree/main/configs/point_rend

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
<MODLE PointRend>使用 COCO2017 数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/#download) 下载。

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
  cd <ModelZoo_path>/PyTorch/contrib/Detection/PointRend/run_scripts
  ```
2. 运行训练。该模型支持单机单卡。
  ```
  python run_PointRend.py --config ../configs/point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py --launcher pytorch --nproc-per-node 1  --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/coco/" "val_dataloader.dataset.data_root=/data/teco-data/coco/"
  ```
  #SDAA设备开启AMP训练时损失会出现训练过慢，损失出现nan现象，关闭AMP可正常训练，可对齐
    更多训练参数参考 run_scripts/argument.py
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.11007853406138651
MeanAbsoluteError: -0.40244024515151977
Rule,mean_absolute_error -0.40244024515151977
pass mean_relative_error=-0.11007853406138651 <= 0.05 or mean_absolute_error=-0.40244024515151977 <= 0.0002
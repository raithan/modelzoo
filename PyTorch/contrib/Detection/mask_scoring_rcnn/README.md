# Mask Scoring R-CNN  
## 1. 模型概述
Mask Scoring R-CNN 在 Mask R-CNN 基础上增加了一个专门的网络分支：MaskIoU Head。
该分支对每个预测掩码进行质量评分，学习预测掩码与真实掩码的 IoU（交并比）。
这样，模型最终输出的掩码分数更能反映掩码的真实质量。

- 论文链接：https://arxiv.org/abs/1903.00241
- 仓库链接：https://github.com/open-mmlab/mmdetection/tree/main/configs/ms_rcnn

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
<MODLE Mask Scoring R-CNN>使用 COCO2017 数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/#download) 下载。

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
  cd <ModelZoo_path>/PyTorch/contrib/Detection/mask_scoring_rcnn/run_scripts
  ```
2. 运行训练。该模型支持单机单卡。
  ```
  python run_mask_scoring_rcnn.py --config ../configs/ms_rcnn/ms-rcnn_r50_fpn_1x_coco.py --launcher pytorch --nproc-per-node 1  --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/coco/" "val_dataloader.dataset.data_root=/data/teco-data/coco/"
  ```
  #SDAA设备开启AMP训练时损失会出现nan现象，关闭AMP可对齐
    更多训练参数参考 run_scripts/argument.py
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.06743551642719746
MeanAbsoluteError: -0.2205455482006073
Rule,mean_absolute_error -0.2205455482006073
pass mean_relative_error=-0.06743551642719746 <= 0.05 or mean_absolute_error=-0.2205455482006073 <= 0.0002
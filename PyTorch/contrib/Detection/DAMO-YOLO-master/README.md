# DAMO-YOLO-S
## 1. 模型概述
DAMO-YOLO是一个面向工业落地的目标检测框架，兼顾模型速度与精度，其训练的模型效果超越了目前的一众YOLO系列方法，并且仍然保持极高的推理速度。
DAMO-YOLO引入TinyNAS技术，使得用户可以根据硬件算力进行低成本的检测模型定制，提高硬件利用效率并且获得更高精度。

- 论文链接：https://arxiv.org/pdf/2211.15444
- 仓库链接：https://github.com/tinyvision/damo-yolo?tab=readme-ov-file

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
DAMO-YOLO 使用 COCO217 数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/#download) 下载。

#### 2.2.2 处理数据集
COCO文件目录结构具体配置方式可参考：https://github.com/Atten4Vis/ConditionalDETR/blob/main/README.md。
需要在项目文件中修改数据集链接（Sympoliclink），在DAMO-YOLO-master/damo/config/paths_catalog.py中修改链接。


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip install -r requirements.txt
    pip install -e .
    ```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Detection/DAMO-YOLO/run_scripts
    ```
2. 运行训练。该模型支持单机单卡。
    ```
    python -m torch.distributed.launch \
     --nproc_per_node = 4 \
     tools/train.py \
     -f configs <配置文件> 
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

DAMO-YOLO-master/run_scripts/loss.jpg

MeanRelativeError: -0.0028650100895755664
MeanAbsoluteError: -0.007623176574707031
Rule,mean_absolute_error -0.007623176574707031
pass mean_relative_error=-0.0028650100895755664 <= 0.05 or mean_absolute_error=-0.007623176574707031 <= 0.0002
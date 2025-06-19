# ConditionalDETR
## 1. 模型概述
  DETR方法将Transformer编码器和解码器架构应用于目标检测，并取得了良好的性能。

- 论文链接：https://arxiv.org/abs/2108.06152
- 仓库链接：https://github.com/Atten4Vis/ConditionalDETR

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
ConditionalDETR 使用 COCO 数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/#download) 下载。

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
    pip install -r requirements.txt
    ```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Detection/ConditoinalDETR/run_scripts
    ```
2. 运行训练。该模型支持单机单卡训练,支持分布式训练
   ```
     python -m torch.distributed.launch --nproc_per_node=4 --use_env run_ConditionalDETR.py --coco_path /data/teco-data/COCO --output_dir output/conddetr_r50_epoch50 2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练结果（参考使用[loss.py](./run_scripts/loss.py)）: 


MeanRelativeError: 0.009474825174738601
MeanAbsoluteError: 0.2481804084777832
Rule,mean_relative_error 0.009474825174738601
pass mean_relative_error=0.009474825174738601 <= 0.05 or mean_absolute_error=0.2481804084777832 <= 0.0002
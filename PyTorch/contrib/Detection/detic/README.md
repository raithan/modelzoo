# detic
## 1. 模型概述
Detic（DEtecting Everything In Context）是一种基于通用视觉表示的目标检测方法，提出在单阶段检测框架中结合零样本分类器，实现对大量类别的开放世界检测。它利用预训练的图像-文本嵌入（如 CLIP）为每个类别生成语义权重，通过“零样本分类器 + R-CNN”方式预测任意类别目标，同时保留常规检测器的回归能力。Detic 支持 LVIS 等大规模数据集，能在未标注类别上进行检测，显著提升长尾类别的性能，强调语义知识迁移和上下文信息利用，是扩展检测类别的一种高效方法。

- 论文链接：*ECCV 2022 ([arXiv 2201.02605](http://arxiv.org/abs/2201.02605))*
- 仓库链接：https://github.com/open-mmlab/mmdetection/tree/main/projects/Detic_new


## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
1. LVIS dataset is adopted as box-labeled data,  [LVIS](https://www.lvisdataset.org/) is available from official website or mirror.  You need to generate `lvis_v1_train_norare.json` according to the [official prepare datasets](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md#coco-and-lvis) for open-vocabulary LVIS, which removes the labels of 337 rare-class from training. You can also download [lvis_v1_train_norare.json](https://download.openmmlab.com/mmdetection/v3.0/detic/data/lvis/annotations/lvis_v1_train_norare.json) from our backup. The directory should be like this.

    ```shell
    mmdetection
    ├── data
    │   ├── lvis
    │   │   ├── annotations
    │   │   |	├── lvis_v1_train.json
    │   │   |	├── lvis_v1_val.json
    │   │   |	├── lvis_v1_train_norare.json
    │   │   ├── train2017
    │   │   ├── val2017
    ```
2. `data/metadata/` is the preprocessed meta-data (included in the repo). Please follow the [official instruction](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md#metadata) to pre-process the  LVIS dataset. You will generate `lvis_v1_train_cat_info.json` for Federated loss, which contains the frequency of each category of training set of LVIS. In addition, `lvis_v1_clip_a+cname.npy` is the pre-computed CLIP embeddings for each category of LVIS. You can also choose to directly download [lvis_v1_train_cat_info](https://download.openmmlab.com/mmdetection/v3.0/detic/data/metadata/lvis_v1_train_cat_info.json) and [lvis_v1_clip_a+cname.npy](https://download.openmmlab.com/mmdetection/v3.0/detic/data/metadata/lvis_v1_clip_a%2Bcname.npy) form our backup. The directory should be like this.

    ```shell
    mmdetection
    ├── data
    │   ├── metadata
    │   │   ├── lvis_v1_train_cat_info.json
    │   │   ├── lvis_v1_clip_a+cname.npy
    ```
### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip install  -U openmim 
    pip install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
    pip install opencv_python mmcv==2.1.0 --no-deps
    mim install -e .
    pip install -r requirements.txt
    pip install git+https://github.com/lvis-dataset/lvis-api.git
    pip install git+https://github.com/openai/CLIP.git
    ```
### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Detection/detic/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
    python run_detic.py ../projects/Detic_new/detic_centernet2_r50_fpn_4x_lvis_boxsup.py \
    --nnodes 1     --nproc_per_node 1\
    --cfg-options "train_cfg.max_iters=200"  --cfg-options "train_cfg.val_interval=90000" 2>&1 | tee sdaa.log

   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

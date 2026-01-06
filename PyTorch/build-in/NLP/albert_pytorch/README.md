# Albert

## 1. 模型概述
Albert是自然语言处理模型，基于Bert模型修改得到。相比于Bert模型，Albert的参数量缩小了10倍，减小了模型大小，加快了训练速度。在相同的训练时间下，Albert模型的精度高于Bert模型。

- 参考实现: https://github.com/lonePatient/albert_pytorch 

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
用户自行下载 SST-2 ，在模型根目录下创建 dataset 目录，并放入数据集。

#### 2.2.2 获取权重
Official download links: [google albert](https://github.com/google-research/ALBERT)

Adapt to this version，download pytorch model (google drive):

**v1**

- [albert_base_v1.zip](https://drive.google.com/open?id=1dVsVd6j8rCTpqF4UwnqWuUpmkhxRkEie)
- [albert_large_v1.zip](https://drive.google.com/open?id=18dDXuIHXYWibCLlKX5_rZkFxa3VSc5j1)
- [albert_xlarge_v1.zip](https://drive.google.com/open?id=1jidZkLLFeDuQJsXVtenTvV_LU-AYprJn)
- [albert_xxlarge_v1.zip](https://drive.google.com/open?id=1PV8giuCEAR2Lxaffp0cuCjXh1tVg7Vj_)

**v2**

- [albert_base_v2.zip](https://drive.google.com/open?id=1byZQmWDgyhrLpj8oXtxBG6AA52c8IHE-)
- [albert_large_v2.zip](https://drive.google.com/open?id=1KpevOXWzR4OTviFNENm_pbKfYAcokl2V)
- [albert_xlarge_v2.zip](https://drive.google.com/open?id=1W6PxOWnQMxavfiFJsxGic06UVXbq70kq)
- [albert_xxlarge_v2.zip](https://drive.google.com/open?id=1o0EhxPqjd7yRLIwlbH_UAuSAV1dtIXBM)

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
    cd <ModelZoo_path>/PyTorch/build-in/NLP/Albert
    ```

2. 运行训练。该模型支持单机单卡。

    ```
    export TORCH_SDAA_AUTOLOAD=cuda_migrate  #自动迁移环境变量
    export SDAA_VISIBLE_DEVICES=0,1,2,3

    CURRENT_DIR=`pwd`
    export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/albert_large_v2
    export DATA_DIR=$CURRENT_DIR/dataset
    export OUTPUR_DIR=$CURRENT_DIR/outputs
    TASK_NAME="sst-2"
    python run_classifier.py \
        --model_type=albert \
        --model_name_or_path=$BERT_BASE_DIR \
        --task_name=$TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir=$DATA_DIR/${TASK_NAME}/ \
        --max_seq_length=128 \
        --per_gpu_train_batch_size=16 \
        --per_gpu_eval_batch_size=8 \
        --spm_model_file=${BERT_BASE_DIR}/30k-clean.model \
        --learning_rate=1e-5 \
        --num_train_epochs=3.0 \
        --logging_steps=4210 \
        --save_steps=4210 \
        --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
        --overwrite_output_dir \
        --seed=42 --max_steps=100
    ```

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./loss.py)）




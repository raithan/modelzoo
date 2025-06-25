# C51

## 1. 模型概述
C51是一种值分布强化学习算法，C51算法的框架依然是DQN算法，采样过程依然使用epsilon-greedy策略取期望贪婪，并且采用单独的目标网络。与DQN算法不同的是，C51算法的卷积神经网络不再是行为值函数，而是支点处的概率，C51算法的损失函数不再是均方而是KL散度。
- 仓库链接：[C51](https://github.com/ShangtongZhang/DeepRL)
- 参考实现：[sdaa基于npu的训练模型参考](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/others/C51)
- 开源代码引入：public_address_statement.md
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

    #安装baselines
    git clone https://github.com/openai/baselines.git
    cd baselines
    pip install -e '.[all]'

    #安装mpi4py
    conda install mpi4py
    ```
    >注意：为成功安装baselines，请确保tensorflow版本大于1.14

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/build-in/others/C51
    ```

2. 运行训练。该模型支持单机单核组。

    ```
    export TORCH_SDAA_AUTOLOAD=cuda_migrate  #自动迁移环境变量
    python3 train_c51.py --use_device='use_gpu' --device_id=0 --max_step=100 --log_interval=1

    ```

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./loss.py)）: 





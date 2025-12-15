
# **Band Split Roformer**
## 1. 模型概述  
Band Split Roformer是字节跳动人工智能实验室开发的用于音乐源分离的 SOTA 注意力网络。他们以大幅优势击败了之前的冠军。该技术利用跨频率（因此是多频带）和时间的轴向注意力机制。他们还通过实验证明，旋转位置编码比学习到的绝对位置有显著的提升。
> **论文链接**：https://arxiv.org/pdf/2309.02612  
> **仓库链接**：https://github.com/ZFTurbo/Music-Source-Separation-Training  

## 2. 快速开始  
使用本模型执行训练的主要流程如下：  
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。  
2. 获取数据集：介绍如何获取训练所需的数据集。  
3. 构建环境：介绍如何构建模型运行所需要的环境。  
4. 启动训练：介绍如何运行训练。  

### 2.1 基础环境安装  

请参考基础环境安装章节，完成训练前的基础环境检查和安装。  

### 2.2 准备数据集  
> 下载数据集到指定文件夹：```/data/teco-data/MUSDB18/```  
> 数据集下载链接：https://zenodo.org/records/1117372/files/musdb18.zip?download=1   
> 解压数据集：``` unzip musdb18.zip```
> 音乐源分离：```python mp42wav.py```   


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境  
1. 执行以下命令，启动虚拟环境。  
    ```
    conda activate torch_env  
    ```
2. 安装python依赖  
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Segmentation/bs_roformer
    apt-get install portaudio19-dev
    pip install -r requirements.txt
    ```
### 2.4 启动训练  
1. 在构建好的环境中，进入训练脚本所在目录。  
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Segmentation/bs_roformer/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。

    -  单机单卡
    ```
    python run_bs_roformer.py     --model_type bs_roformer    --config_path configs/config_musdb18_bs_roformer.yaml     --results_path results/     --data_path '/data/teco-data/MUSDB18-wav/train'     --valid_path /data/teco-data/MUSDB18-wav/test     --num_workers 4     --device_ids 0 2>&1 | tee sdaa.log
   ```
    更多训练参数参考[README](run_scripts/README.md)

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 
![训练loss曲线](./run_scripts/loss.jpg)

MeanRelativeError: 0.012818318475866224
MeanAbsoluteError: -0.17328822910785674
Rule,mean_absolute_error -0.17328822910785674
pass mean_relative_error=0.012818318475866224 <= 0.05 or mean_absolute_error=-0.17328822910785674 <= 0.0002

# FAN

## 1. 模型概述
FAN是一个目标对齐检测网络，通过对目标标志的检测，既能够检测2D也能够检测3D坐标中的点。
- 仓库链接：[FAN](https://github.com/1adrianb/face-alignment)
- 参考代码：[https://github.com/hhj1897/fan_training](https://github.com/hhj1897/fan_training)

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 获取数据集
- 获取数据集。
  用用户自行获取原始数据集[下载链接](https://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz)，可选用的开源数据集包括300-W等，
  进入到项目根目录并创建`data`目录，将数据集放在`data`下：

   ```
   # $FAN_ROOT 为项目根目录
   $FAN_ROOT/data/
   ```

  以300-w数据集为例，数据集目录结构参考如下所示。

   ```
    dataset
       └── 300W
           ├── aww
           ├── bug
           ├── Helen
           │   ├── trainset
           │   └── testset
           └── lfpw
               ├── trainset
               └── testset
   ```
  
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
    cd <ModelZoo_path>/PyTorch/build-in/Detection/FAN
    ```
2. 由于TecoPyTorch api `convolution_backward`默认使用硬件的ACE单元，会转成half来计算，精度有可能不够，导致模型无法收敛。
   需要使用SIMD单元来开启纯fp32计算，训练前添加环境变量：
    ```
    export TORCH_SDAA_CONV2D_BACKWARD_USE_FP32=1
    ```
3. 运行训练脚本。

   该模型支持单机单卡训练。

   - 启动单卡训练。
     ```
     TORCH_SDAA_CONV2D_BACKWARD_USE_FP32=1 python main.py
     ```

   训练完成后，权重文件保存在`checkpoint/`路径下，并输出模型训练精度和性能信息。

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[plot.py](./plot.py)，日志在`logs/`目录下）:<br>
MeanRelativeError: 0.014647970488368241
MeanAbsoluteError: 9.989399286742145e-06
Rule,mean_absolute_error 9.989399286742145e-06
pass mean_relative_error=np.float64(0.014647970488368241) <= 0.05 or mean_absolute_error=np.float64(9.989399286742145e-06) <= 0.0002

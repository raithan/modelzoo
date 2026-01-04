# CenterNet2

## 1. 模型概述
CenterNet2 是一种基于 CenterNet 的目标检测模型，它改进了原有的架构以提供更高效和准确的目标检测能力。该模型通过关键点估计来定位物体的中心，并回归出物体的尺寸和其他属性，从而实现对图像中多个物体的精确定位与分类。此外，CenterNet2 引入了额外的优化和模块，如使用更高分辨率的特征图、改进的后处理算法等，以提升检测性能，尤其在小物体检测和密集场景下表现更加出色。它是计算机视觉领域内用于目标检测任务的一种强有力的方法。


- 参考实现：
    ```
    url=https://github.com/xingyizhou/CenterNet2
    commit_id=bfc8b72e4e2a1612b4aa4c12f7b3db9cdd7b34c4
    ```


## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 准备数据集

- 您需要使用到coco数据集，可以通过[MSCOCO数据集官网](http://mscoco.org/)自行下载数据集，并按照以下目录组织数据集。或者用户进入到根目录，执行以下命令，下载coco数据集。coco数据集包括了图片，labels，annotations。下载完成后数据集默认存在在根目录的data文件中。

    ```
    wget -c http://images.cocodataset.org/zips/train2017.zip
    unzip train2017.zip
    wget -c http://images.cocodataset.org/zips/val2017.zip
    unzip val2017.zip
    wget -c http://images.cocodataset.org/zips/test2017.zip
    unzip test2017.zip
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip annotations_trainval2017.zip
    ```

- 请你按以下结构组织coco数据集：

   ```
    ├── coco
        ├── annotations
            ├──instances_train2017.json
        ├── labels
            ├──train2017
            ├──train2017.cache3
            ├──val2017
            ├──val2017.cache3 
        ├── train2017
        ├── val2017
        ├── test2017
        ├── test-dev2017.txt
        ├── train2017.txt
        ├── train2017.cache
        ├── val2017.cache
        ├── val2017.txt
        ├── LICENSE
        ├── README.md
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Detection/CenterNet2
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_CenterNet2`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_CenterNet2
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_CenterNet2 -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_CenterNet2 /bin/bash
    ```

#### 2.3.5 执行以下命令，启动虚拟环境。

    ```
    conda activate torch_env_py310
    ```

#### 2.3.6 执行以下命令，安装其他环境依赖包。

    ```
    pip install -r requirements.txt
    ```

#### 2.3.7 执行以下命令，安装detectron2。

    ```
    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron2
    pip install -e .
    ```

 - 检查detectron2是否安装成功

  ```
  python -c "import detectron2; print(detectron2.__version__)"
  ```

  输出结果如下则安装成功：
  0.6

 - 调整detectron2目录结构
  请将<modelzoo-dir>/PyTorch/Detection/CenterNet2/detectron2/detectron2 to <modelzoo-dir>/PyTorch/Detection/CenterNet2/detectron2/
  若命名冲突请在调整前将 <modelzoo-dir>/PyTorch/Detection/CenterNet2/detectron2/ 重新命名

 - 调整detectron2代码细节
    1. 修改detectron2/detectron2/utils/env.py中大约45行的代码
        ```    
        torch.cuda.manual_seed_all(str(seed))
        ```
        修改为：
        ```        
        torch.cuda.manual_seed_all(seed)
        ```               
    2. 修改detectron2/detectron2/utils/collect_env.py中大约147行的代码
        ```
        if has_gpu:
            devices = defaultdict(list)
            for k in range(torch.cuda.device_count()):
                cap = ".".join((str(x) for x in torch.cuda.get_device_capability(k)))
                name = torch.cuda.get_device_name(k) + f" (arch={cap})"
                devices[name].append(str(k))
            for name, devids in devices.items():
                data.append(("GPU " + ",".join(devids), name))
        ```
        修改为：
        ```
        if has_gpu:
            devices = []
            if torch.cuda.is_available():
                for k in range(torch.cuda.device_count()):
                    try:
                        cap = ".".join((str(x) for x in torch.cuda.get_device_capability(k)))
                    except Exception:
                        cap = "unknown"
                    name = torch.cuda.get_device_name(k)
                    devices.append(f"{name} ({cap})")
            else:
                devices.append("Non-CUDA device (capability not available)")
            data.append(("GPU(s)", "\n".join(["- " + dev for dev in devices])))
        ```
    
 - 验证detectorn2安装
        ```
        cd detectron2/demo
        python demo/predictor.py --config-file ../configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml --input <your-image-path>
        ```

### 2.4 启动训练

#### 2.4.1 在Docker环境中，进入训练脚本所在目录。
    ```
    cd /workspace/Detection/CenterNet2
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 您需要从[CenterNet2_R50_1x.pth](https://drive.google.com/file/d/1Qn0E_F1cmXtKPEdyZ_lSt-bnM9NueQpq)下载预训练的权重，并请放置在 '/workspace/Detection/CenterNet2models/CenterNet2_R50_1x.pth' 路径下
  
  - 启动训练：
    ```
    python train_net.py --config-file configs/CenterNet2_R50_1x.yaml
    ```
  - 开启混合精度amp训练：
    ```
    python train_amp.py --config-file configs/CenterNet2_R50_1x.yaml
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: -0.0037199840295415246
MeanAbsoluteError: -0.010466101694915264
Rule,mean_absolute_error -0.010466101694915264
pass mean_relative_error=-0.0037199840295415246 <= 0.05 or mean_absolute_error=-0.010466101694915264 <= 0.0002


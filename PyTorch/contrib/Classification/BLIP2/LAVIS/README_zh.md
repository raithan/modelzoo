# BLIP2
## 1. 模型概述
是一种通用且高效的预训练策略，可以轻松收获用于视觉语言预训练的预训练视觉模型和大型语言模型（LLM）的开发。BLIP-2在零样本 VQAv2（65.0 对 56.3）上击败了Flamingo，在零样本字幕方面建立了新的最先进技术（NoCaps 121.6 CIDEr分数与之前的最佳成绩 113.2）。BLIP-2配备了强大的LLM（例如 OPT、FlanT5），还为各种有趣的应用程序解锁了新的零样本指令视觉到语言生成功能！

- 论文链接：[[2301.12597\]]BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models(https://arxiv.org/abs/2301.12597)
- 仓库链接：https://github.com/salesforce/LAVIS/tree/main/run_scripts/blip2/
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
BLIP2使用 COCO2014 数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/) 下载。

### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
    pip install salesforce-lavis
    ### 或按照 LAVIS 说明从源代码安装。
    cd LAVIS
    pip install -e .
    pip install -r requirements.txt
    ```
### 2.4 启动训练
针对BLIP2模型的图像标题进行微调。
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/BLIP2/LAVIS
    ```

2. 直接运行训练。该模型支持单机单卡。
    ```
   python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/blip2/train/caption_coco_ft.yaml --options datasets.coco_caption.build_info.images.storage="$data_path" 2>&1 |tee sdaa.log

   ```
    更多训练参数参考 train.py 和caption_coco_ft.yaml。

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](loss.py)）: 

MeanRelativeError: -8.57392649890155e-05
MeanAbsoluteError: -0.00018689776499999312
Rule,mean_absolute_error -0.00018689776499999312
pass mean_relative_error=-8.57392649890155e-05 <= 0.05 or mean_absolute_error=-0.00018689776499999312 <= 0.0002
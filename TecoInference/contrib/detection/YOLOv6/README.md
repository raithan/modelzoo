# YOLOv6

## 1. 模型概述

YOLOv6 是一款单阶段实时目标检测模型，兼顾速度与精度，适合工业部署场景。本文档给出在 Tecorigin 硬件/推理栈（SDAA/TecoDNN/TecoInfer）上的 **从环境到推理** 的完整指引，并按验收标准提供：**单样本推理、文件夹推理、数据集推理** 三部分。

---

## 2. 快速开始

使用本模型执行推理的主要流程如下：

1. **基础环境安装**：完成推理前的环境检查与安装。
2. **安装第三方依赖**：安装推理脚本所需 Python 依赖。
3. **获取 ONNX 文件**：从 YOLOv6 权重导出 ONNX 模型。
4. **获取数据集**：准备 COCO 数据集（或使用示例图片）。
5. **启动推理**：单样本 / 文件夹 / 数据集推理。
6. **精度验证**：验证 COCO mAP（本节命令提供，结果占位，稍后填写）。

### 2.1 基础环境安装

请参考推理首页的**基础环境安装**章节，完成以下检查：

* 已安装并加载 **SDAA Driver/Runtime、TecoDNN、TecoBLAS、TecoCustom** 等组件；
* Conda 环境可用（本文示例默认环境名：`yolov12`）；
* 已安装 **TecoInference** 运行时（Python API `engine.tecoinfer_pytorch.TecoInferEngine` 可被导入）。

> 提示：如果此前已完成 YOLOv12/YOLOv6 的环境配置，可直接复用，无需重复安装。

### 2.2 安装第三方依赖

1. 进入 Conda 环境：

```bash
conda activate tvm-build_py310
```

2. 进入推理脚本所在目录（本文以 `<modelzoo_dir>/TecoInference/example/detection/yolov6` 为例；如你已将脚本拷贝到当前工作目录，直接在当前目录执行下列命令即可）：

```bash
cd <modelzoo_dir>/TecoInference/example/detection/yolov6
```

3. 安装依赖（若存在 `requirements.txt`）：

```bash
pip install -r requirements.txt
```

> 如未提供 `requirements.txt`，可按需安装：`opencv-python-headless`, `tqdm`, `pycocotools`（做 COCO 评测需要），并确保 `numpy==1.26.x` 以避免与旧版 PyTorch/导出脚本冲突。

### 2.3 获取 ONNX 文件

你可以从 `yolov6n.pt` 导出 ONNX（或直接使用现成的 `yolov6n.onnx`）。推荐导出命令：

```bash
# 建议先确保 numpy<2
pip install --force-reinstall "numpy==1.26.4"

# 在 YOLOv6 源码目录执行（meituan/YOLOv6）
python deploy/ONNX/export_onnx.py \
  --weights yolov6n.pt \
  --img 640 \
  --batch 1 \
  --simplify

# 导出成功后得到 yolov6n.onnx（及简化版本）
```

> 说明：若你已具备 `yolov6n.onnx`，可直接进入下一步。

### 2.4 获取数据集（COCO）

本文以 **COCO** 为例，假设数据集路径为：`/data/teco-data/coco`，其下至少包含 `val2017/` 与 `annotations/instances_val2017.json`。

为配合数据集推理脚本，请生成 `val2017.txt`（每行一个图像绝对路径或相对路径）：

```bash
find /data/teco-data/coco/val2017 -type f -name "*.jpg" | sort > /data/teco-data/coco/val2017.txt
```

> 如需文件夹推理演示，可抽样 10 张图片：

```bash
mkdir -p imgs
find /data/teco-data/coco/val2017 -type f -name "*.jpg" | shuf -n 10 | xargs -I{} cp "{}" imgs/
```

### 2.5 启动推理

以下命令基于随文的三个推理脚本：`example_single_batch.py / example_multi_batch.py / example_valid.py`。

#### 2.5.1 单样本推理

```bash
python example_single_batch.py \
  --ckpt yolov6n.onnx \
  --batch-size 1 \
  --target sdaa
```

> 结果图片可通过脚本开关 `--save_result True` 保存（若脚本支持）。

#### 2.5.2 文件夹推理

```bash
python example_multi_batch.py \
  --ckpt yolov6n.onnx \
  --batch-size 1 \
  --target sdaa \
  --data-path ./imgs
```


#### 2.5.3 数据集推理（COCO）

```bash
python example_valid.py \
  --model_name yolov6n \
  --data-path /data/teco-data/coco \
  --ckpt yolov6n.onnx \
  --batch-size 4 \
  --shape 640 \
  --target sdaa \
  --half True
```

> 说明：`--half True` 表示使用 FP16；若当前仅 CPU 或 EP 不支持半精度，请改为 `--half False`。

#### 模型推理参数说明（常用）

| 参数           | 说明                              | 默认值                                                               |
| ------------ | ------------------------------- | ----------------------------------------------------------------- |
| `data-path`  | 图片/数据集路径；数据集评测需包含 `val2017.txt` | 单图：`./imgs/bus.jpg`；多图：`./imgs/coco/`；数据集：`/data/teco-data/coco/` |
| `ckpt`       | ONNX 模型路径                       | `./yolov6n.onnx`                                                  |
| `batch-size` | 推理批大小                           | `1`（建议先从 1 验证）                                                    |
| `shape`      | 推理输入尺寸                          | `640`                                                             |
| `target`     | 设备后端                            | `sdaa`（或 `cpu`）                                                   |
| `input_name` | ONNX 输入名                        | `images`                                                          |
| `half`       | 是否使用 FP16                       | `True`（EP 不支持时请设 `False`）                                         |
| `conf-thres` | 置信度阈值                           | `0.25`（单/多图）；`0.001`（COCO 评测）                                     |
| `iou-thres`  | NMS IoU 阈值                      | `0.45`（单/多图）；`0.65`（COCO 评测）                                      |
| `max-det`    | 每图最大检测数                         | `1000`（单/多图）；`300`（COCO 评测）                                       |

> 其余参数请查看脚本内 `argparse` 注释（如 `model_name`, `pass_path`, `save_result` 等）。

### 2.6 精度验证

请先确保 COCO 验证集与 `val2017.txt` 就绪。运行：

```bash
python example_valid.py \
  --model_name yolov6n \
  --data-path /data/teco-data/coco \
  --ckpt yolov6n.onnx \
  --batch-size 4 \
  --shape 640 \
  --target sdaa \
  --half True
```

**精度结果如下**：
请提前准备好coco数据集，执行以下命令，获得推理精度数据。
```
python example_valid.py --model_name yolov6n --data-path /data/teco-data/coco  --ckpt yolov6n.onnx --batch-size 4 --shape 640 --target sdaa --half True
```
精度结果如下：
```
mAP@0.5~0.95:  0.6935122032657104
summary: avg_sps: 15.558954579681073, e2e_time: 319.1244738101959, avg_inference_time: 0.011535129794625856, avg_preprocess_time: 0.003789523499077472, avg_postprocess: 0.24244873184764906

```
结果说明：

参数	说明
avg_sps	吞吐量(images/s)
e2e_time	端到端总耗时(s)
avg_inference_time	平均推理计算时间(s)
avg_preprocess_time	平均预处理时间(s)
avg_postprocess	平均后处理时间(s)
mAP@0.5~0.95	数据集验证精度
---

## 3. 目录结构示例（参考）

```
example/detection/yolov6/
├── example_single_batch.py   # 单图推理
├── example_multi_batch.py    # 文件夹推理
├── example_valid.py          # COCO 评测
├── yolov6n.onnx              # 导出的 ONNX（示例）
└── imgs/                     # 测试图片（可选）
```

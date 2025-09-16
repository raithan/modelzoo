# YOLOv12 在 SDAA/TecoInfer 上的部署与推理指南

> 适用环境：`tvm-build_py310` conda 环境，TecoInference 2.1.x，SDAA/TecoDNN/TecoBLAS/TecoCustom 2.1.x。

---

## 1. 模型概述

YOLOv12 是 Ultralytics 系列的轻量检测模型。本指南基于 **Tecorigin 推理栈（SDAA + TecoDNN + TecoInfer）**，给出从 **环境准备 → ONNX 导出 → 单图/文件夹/数据集推理** 的完整步骤，并整理了常见问题排查与优化建议。

---

## 2. 快速开始（TL;DR）

```bash
# 0) 进入环境
conda activate tvm-build_py310

# 1) 安装（或修复）导出依赖
pip install -U ultralytics huggingface_hub safetensors onnx onnxsim
# 若遇到 NumPy 2.x 兼容问题，建议：
pip install --force-reinstall "numpy==1.26.4"

# 2) 导出 ONNX（在 yolov12 目录下）
PYTHONPATH=..:$PYTHONPATH \
python ONNX/export_onnx12.py \
  --weights ../yolov12n.pt \
  --img-size 640 640 \
  --batch-size 1 \
  --dynamic-batch \
  --simplify \
  --fp16_post
# 看到日志包含：Preprocess packed: BGR->RGB + normalize. / Model converted to FP16

# 3) 单图推理（在 yolov12 目录下）
python example_single_batch.py \
  --ckpt ./yolov12n.onnx \
  --data-path ./imgs/bus.jpg \
  --shape 640 \
  --target sdaa \
  --half true \
  --model_name yolov12n \
  --input_name images \
  --onnx-has-preprocess \
  --names coco \
  --conf-thres 0.001 \
  --iou-thres 0.6 \
  --save_result true
# 期望输出：Saved: output_image.jpg
```

---

## 3. 环境与依赖

### 3.1 基础组件检查

执行脚本时会自动打印版本，例如：

```
Teco-infer  | 2.1.0rc0+gitxxxxx
TecoDNN     | 2.1.0 (/opt/tecoai/lib64/libtecodnn.so)
TecoBLAS    | 2.1.0 (/opt/tecoai/lib64/libtecoblas.so)
TecoCustom  | 1.20.0a0 (/opt/tecoai/lib64/libtecocustom.so)
SDAA Runtime| 2.1.0 (/opt/tecoai/lib64/libsdaart.so)
SDAA Driver | 2.1.0
```

### 3.2 Python 依赖建议

* 导出阶段：`ultralytics`, `huggingface_hub`, `safetensors`, `onnx`, `onnxsim`
* 推理阶段：`opencv-python(-headless)`, `numpy (<=1.26.x 建议)`, `tqdm`
* 数据集评测：`pycocotools`

> 若 `NumPy 2.x` 触发 “A module compiled for NumPy 1.x …” 等错误，请降级到 `1.26.4`。

---

## 4. ONNX 导出（YOLOv12 → ONNX）

在 `contrib/detection/yolov12/ONNX` 下执行（确保能 import 本地 ultralytics）：

```bash
PYTHONPATH=..:$PYTHONPATH \
python export_onnx12.py \
  --weights ../yolov12n.pt \
  --img-size 640 640 \
  --batch-size 1 \
  --dynamic-batch \
  --simplify \
  --fp16_post
```

**导出要点**

* 建议 `--fp16_post`（导出后转 FP16），与推理侧 `--half true` 对齐。
* 日志若出现 `Preprocess packed`，说明 ONNX **已内置 BGR->RGB + Normalize**，推理脚本需加 `--onnx-has-preprocess`。
* 若报 `No module named 'huggingface_hub'`，请先 `pip install -U huggingface_hub`。
* 若报 `scaled_dot_product_attention` / `qk` 等错误，多为包版本/仓库混用问题，使用本目录自带 `export_onnx12.py` 可规避。

---

## 5. 推理脚本使用

目录（示例）：

```
contrib/detection/yolov12/
├── example_single_batch.py   # 单图推理（已适配 YOLOv12）
├── example_multi_batch.py    # 文件夹推理
├── example_valid.py          # COCO 验证评测
└── yolov12n.onnx             # 导出模型
```

### 5.1 单样本推理

```bash
python example_single_batch.py \
  --ckpt ./yolov12n.onnx \
  --data-path ./imgs/bus.jpg \
  --shape 640 \
  --target sdaa \
  --half true \
  --model_name yolov12n \
  --input_name images \
  --onnx-has-preprocess \
  --names coco \
  --conf-thres 0.001 \
  --iou-thres 0.6 \
  --save_result true
```

输出图片：`output_image.jpg`。

### 5.2 文件夹推理

```bash
python example_multi_batch.py \
  --ckpt ./yolov12n.onnx \
  --data-path ./imgs \
  --shape 640 \
  --target sdaa \
  --half true \
  --model_name yolov12n \
  --input_name images \
  --onnx-has-preprocess \
  --names coco \
  --batch-size 1 \
  --save_result true
```

### 5.3 数据集推理（COCO 验证）

1. 安装评测依赖：

```bash
pip install -U pycocotools
```

2. 准备数据：`/data/teco-data/coco`，包含 `val2017/` 与 `annotations/instances_val2017.json`。
3. 运行：

```bash
python example_valid.py \
  --ckpt ./yolov12n.onnx \
  --data-path /data/teco-data/coco \
  --batch-size 32 \
  --shape 640 \
  --target sdaa \
  --half true \
  --model_name yolov12n \
  --input_name images \
  --conf-thres 0.001 \
  --iou-thres 0.6 \
  --workers 8 \
  --save-json
```

* 若 `pycocotools` 未安装会跳过 mAP 打印，仅保存 `*_predictions.json`。
* 安装后将显示 `mAP@0.5:0.95` 等指标与吞吐/时延统计。

---

## 6. 常见问题与修复

* **ImportError: huggingface\_hub / ultralytics** → `pip install -U ultralytics huggingface_hub safetensors`。
* **NumPy 2.x 兼容报错**（导出阶段）→ `pip install --force-reinstall "numpy==1.26.4"`。
* **SDAA invalid argument**（构图或输入不匹配）→ 确认 `--shape` 与导出一致；若 ONNX 内置预处理，运行侧必须加 `--onnx-has-preprocess`；`input_name` 与 ONNX 名称对齐（通常 `images`）。
* **推理无框/无结果** → 降低 `--conf-thres`（如 `0.001`）、确认 `--names coco` 以适配类名索引；也可先关闭 `--half` 排除数值差异。
* **example\_valid 不打印精度** → 安装 `pycocotools`，或开启 `--save-json` 后手动评测。

---

## 7. 性能与精度建议

* **精度对齐**：评测时使用 `--conf-thres 0.001`、`--iou-thres 0.65` 更接近通用评测设置；确保 `val2017.txt` 或脚本的遍历逻辑正确。
* **性能**：SDAA 上建议使用 `--half true` 与 **动态 batch** 导出；多图推理时 `--batch-size` 视显存与算力调整。
* **可重复性**：固定 `img-size=640`、`input_name=images`、`names=coco`，并在日志中保留 TecoInfer/SDAA 版本打印。

---

## 8. 目录结构参考

```
contrib/detection/yolov12/
├── ONNX/
│   └── export_onnx12.py
├── example_single_batch.py
├── example_multi_batch.py
├── example_valid.py
├── imgs/
│   └── bus.jpg
└── yolov12n.onnx
```

---

## 9. 复现脚本（一键顺序）

```bash
conda activate tvm-build_py310
pip install -U ultralytics huggingface_hub safetensors onnx onnxsim opencv-python-headless tqdm
pip install --force-reinstall "numpy==1.26.4"  # 如遇导出兼容问题

cd contrib/detection/yolov12
PYTHONPATH=..:$PYTHONPATH \
python ONNX/export_onnx12.py \
  --weights ../yolov12n.pt --img-size 640 640 --batch-size 1 --dynamic-batch --simplify --fp16_post

python example_single_batch.py \
  --ckpt ./yolov12n.onnx --data-path ./imgs/bus.jpg --shape 640 \
  --target sdaa --half true --model_name yolov12n --input_name images \
  --onnx-has-preprocess --names coco --conf-thres 0.001 --iou-thres 0.6 --save_result true

# 数据集评测（可选）
pip install -U pycocotools
python example_valid.py \
  --ckpt ./yolov12n.onnx --data-path /data/teco-data/coco --batch-size 32 \
  --shape 640 --target sdaa --half true --model_name yolov12n --input_name images \
  --conf-thres 0.001 --iou-thres 0.6 --workers 8 --save-json
```

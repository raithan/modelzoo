# RMBG-1.4 部署与训练指南（SDAA 版本）

> 本文档说明如何在新型显卡环境下，使用 **SDAA** 后端部署与训练 RMBG-1.4（或任意抠图/背景移除模型）。默认操作系统为 Linux，Python ≥ 3.9。

---

## 目录

* [项目简介](#项目简介)
* [环境准备](#环境准备)
* [依赖安装](#依赖安装)
* [数据准备](#数据准备)
* [快速开始：100 步训练示例](#快速开始100-步训练示例)
* [日志与损失可视化](#日志与损失可视化)
* [推理与导出](#推理与导出)
* [常见问题](#常见问题)
* [目录结构参考](#目录结构参考)

---

## 项目简介

RMBG-1.4 是一种通用的人像/前景背景移除模型。本文档提供：

1. 在 **SDAA** 设备上运行与训练的完整流程。
2. 简化版本训练脚本（`train_rmbg_sdaa_final.py`）。
3. 损失对比与可视化脚本（`compare_loss_vis.py`）。

---

## 环境准备

1. 安装并配置 SDAA 驱动、Runtime 与相关库（如 `libsdaart.so`、`libtecodnn.so` 等）。
2. 使用 Conda 或 venv 创建独立环境：

```bash
conda create -n rmbg_sdaa python=3.10 -y
conda activate rmbg_sdaa
```

3. 安装与 SDAA 匹配的 PyTorch 与 torch-sdaa（使用厂商提供的 whl 包或镜像源）：

```bash
pip install torch-2.4.0a0+git4451b0e-cp310-cp310-linux_x86_64.whl \
            torch_sdaa-2.1.0-cp310-cp310-linux_x86_64.whl
pip install torchvision==0.19.0
```

4. 其他依赖：

```bash
pip install -U timm einops pillow opencv-python tqdm pyyaml scikit-image safetensors matplotlib scipy
# 强烈建议使用 numpy<2
pip install "numpy==1.26.4" --upgrade --force-reinstall
```

---

## 数据准备

假设使用 **PPHumanSeg** 或其他已标注前景/掩码的数据集。

要求的数据结构：

```
/data/teco-data/PPHumanseg/
  ├── train/
  │   ├── images/*.jpg|png
  │   └── masks/*.png
  └── val/
      ├── images/*.jpg|png
      └── masks/*.png
```

> 文件名需一一对应（同 stem）。若源数据为 `mini_supervisely/Images` + `mini_supervisely/Annotations`，请先用脚本拷贝并重命名到上述结构。

示例拷贝脚本（匹配 Images/Annotations）：

```python
# copy_pairs_pphumanseg.py
import shutil, pathlib, random
root = pathlib.Path("/data/teco-data/PPHumanseg/mini_supervisely")
imgs = list((root/"Images").glob("*.*"))
anns = {p.stem: p for p in (root/"Annotations").glob("*.png")}
random.shuffle(imgs)
tr, vl = int(len(imgs)*0.9), len(imgs)
for subset, pairs in [("train", imgs[:tr]), ("val", imgs[tr:vl])]:
    im_out = pathlib.Path(f"/data/teco-data/PPHumanseg/{subset}/images"); im_out.mkdir(parents=True, exist_ok=True)
    mk_out = pathlib.Path(f"/data/teco-data/PPHumanseg/{subset}/masks");  mk_out.mkdir(parents=True, exist_ok=True)
    for im in pairs:
        m = anns.get(im.stem)
        if m:
            shutil.copy(im, im_out/(im.stem+im.suffix.lower()))
            shutil.copy(m, mk_out/(im.stem+".png"))
```

---

## 快速开始：100 步训练示例

使用提供的简化脚本 `train_rmbg_sdaa_final.py`：

```bash
python train_rmbg_sdaa_final.py \
  --train_root /data/teco-data/PPHumanseg/train \
  --val_root   /data/teco-data/PPHumanseg/val \
  --max-steps 100 \
  --batch-size 8 \
  --log-file /data/bigc-data/zh/cog-RMBG/sdaa_loss.log \
  --save-dir weights \
  --amp
```

脚本特性：

* 自动检测并使用 SDAA 设备（否则退回 CPU）。
* 单进程 DataLoader，稳定不挂起。
* 采用 BCEWithLogits + Dice Loss。
* 每步写入 `sdaa_loss.log`，格式：`step,loss`。
* 每 20 步验证并保存最优权重到 `weights/`。

> 如需加载 safetensors 权重，增加 `--load-ckpt /path/to/model.safetensors`。

---

## 日志与损失可视化

使用 `compare_loss_vis.py` 对比 SDAA 与基准设备的损失：

```bash
python compare_loss_vis.py \
  --sdaa-log sdaa_loss.log \
  --cuda-log cuda_loss.log \
  --align truncate \
  --smooth-window 5 \
  --smooth-poly 1 \
  --out-png loss_cmp.png \
  --out-metrics loss_metrics.txt
```

输出内容：

* **MeanRelativeError** / **MeanAbsoluteError**
* 对比曲线图 `loss_cmp.png`
* 指标记录 `loss_metrics.txt`

> 若日志格式不同（如含 `rank:0 train.loss:`），脚本会自动匹配解析；也可自行改正则表达式。

---

## 推理与导出

简单推理脚本示例（伪代码）：

```python
from PIL import Image
import torch
from model import RMBGNet  # 替换为真实模型

device = torch.device('sdaa:0' if hasattr(torch, 'sdaa') else 'cpu')
model = RMBGNet(...).to(device)
model.load_state_dict(torch.load('weights/best_step100_*.pth', map_location=device))
model.eval()

img = Image.open('test.jpg').convert('RGB')
# 预处理 -> tensor -> device
alpha_logit = model(tensor)
alpha = alpha_logit.sigmoid().cpu().numpy()
# 保存 alpha / 抠图结果
```

导出 ONNX：

```bash
python export_onnx.py --weights weights/best.pth --out rmbg14.onnx
```

> 需确保导出脚本不包含仅在训练中使用的模块。

---

## 常见问题

1. **`unsupported pickle protocol: 51`**
   使用 `safetensors` 加载或 `torch.load(..., weights_only=True)`，避免旧版 pickle。

2. **NumPy 报 `_ARRAY_API not found` 或模块不兼容**
   使用 `numpy==1.26.4`。

3. **DataLoader 卡死/无输出**
   先将 `num_workers=0`，`pin_memory=False`，并关闭混精（AMP）；跑通后再逐步加回。

4. **日志没写入**
   使用绝对路径；或在脚本中提前 `open(..., buffering=1)` 并周期性 `flush()`。

---

## 目录结构参考

```
project/
├── train_rmbg_sdaa_final.py     # 训练脚本
├── compare_loss_vis.py          # 损失对比与可视化
├── export_onnx.py               # 可选：导出 ONNX
├── weights/                     # 保存权重
├── sdaa_loss.log                # SDAA 训练损失日志
├── cuda_loss.log                # 基准设备损失日志（可选）
└── datasets/                    # 数据根目录（可选）
```

---

如需：

* 替换成 RMBG-1.4 官方模型结构；
* 增加多卡分布式训练；
* 添加更多指标（RMSE、最大误差等）；
* 可视化结果（抠图前后效果对比）；

请在 Issue 或邮件中反馈，我们会继续完善。

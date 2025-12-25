# DeiT-Base 训练与损失对比（SDAA 平台）

> **目标**：在 SDAA GPU 平台上训练 DeiT-Base，并记录前 100 步损失；随后用脚本对比与另一份基准日志（如 A100、V100 或其他平台）之间的误差并可视化。

---

## 目录

* [项目结构](#项目结构)
* [环境准备](#环境准备)
* [数据集准备](#数据集准备)
* [训练脚本说明](#训练脚本说明)
* [快速开始](#快速开始)
* [损失记录与可视化](#损失记录与可视化)
* [日志对齐与误差评估](#日志对齐与误差评估)
* [常见问题](#常见问题)
* [许可证](#许可证)

---

## 项目结构

```
/workspace/deit_sdaa/
├── train.py                 # 训练入口脚本（SDAA 适配版）
├── compare_loss.py          # 损失对比与可视化脚本
├── outputs/
│   ├── test100/
│   │   ├── sdaa_loss.csv    # 训练 100 步时保存的损失
│   │   ├── epoch001.pth     # 示例权重
│   │   └── log.txt          # 可选日志
└── data/
    └── imagenet/            # ImageNet 根目录（含 train/、val/）
```

---

## 环境准备

```bash
conda create -n deit-sdaa python=3.10 -y
conda activate deit-sdaa

# 基础依赖
pip install timm==0.9.16 torchvision tqdm tensorboard pyyaml matplotlib scipy pandas

# 已安装 torch 与 torch_sdaa 的情况下，确保环境变量：
export SDAA_VISIBLE_DEVICES=0   # 单卡示例，根据需要调整
```

> 若使用多卡分布式训练，请额外配置 `TCCL` 通讯相关环境变量（如网卡名）。

---

## 数据集准备

* 使用标准 **ImageNet-1k** 目录结构：

  ```
  /data/teco-data/imagenet/
    ├── train/
    │   ├── n01440764/xxx.jpeg
    │   └── ...
    └── val/
        ├── n01440764/xxx.jpeg
        └── ...
  ```
* 也可以替换为任意自定义分类数据集，保证与 ImageNet 一致的目录格式（类别=子目录）。

---

## 训练脚本说明

`train.py` 关键特性：

* 设备类型：`sdaa`
* 优化器：`AdamW`
* 学习率调度：`CosineAnnealingLR`
* 支持 `--max_steps` 提前终止训练
* 将训练损失逐步写入 CSV/LOG（`--loss_log`）
* 关闭 timm 的 CUDA 专用 PrefetchLoader（`use_prefetcher=False`）

主要参数：

| 参数               | 说明                   | 默认                      |
| ---------------- | -------------------- | ----------------------- |
| `--data`         | 数据根目录（含 train/、val/） | 必填                      |
| `--model`        | timm 中的模型名           | `deit_base_patch16_224` |
| `--batch_size`   | batch 大小             | 64                      |
| `--epochs`       | 训练总 epoch            | 300                     |
| `--lr`           | 初始学习率                | 5e-4                    |
| `--weight_decay` | 权重衰减                 | 0.05                    |
| `--accum_steps`  | 梯度累积步数               | 1                       |
| `--amp`          | 开启 bfloat16 自动混合精度   | 关闭                      |
| `--max_steps`    | 全局步数上限（用于快速测试）       | -1（无限制）                 |
| `--loss_log`     | 损失日志文件名              | `sdaa_loss.csv`         |
| `--out`          | 输出目录                 | `./outputs`             |
| `--dist`         | 启用分布式训练              | 关闭                      |

---

## 快速开始

### 单卡运行 100 步测试

```bash
SDAA_VISIBLE_DEVICES=0 python train.py \
  --data /data/teco-data/imagenet \
  --model deit_base_patch16_224 \
  --batch_size 64 \
  --epochs 1 \
  --lr 5e-4 \
  --amp \
  --max_steps 100 \
  --loss_log sdaa_loss.csv \
  --out ./outputs/test100
```

### 多卡（单机 4 卡示例）

```bash
SDAA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --dist \
  --data /data/teco-data/imagenet \
  --model deit_base_patch16_224 \
  --batch_size 64 \
  --epochs 300 \
  --lr 5e-4 \
  --amp \
  --loss_log sdaa_loss.csv \
  --out ./outputs/deit_base_sdaa
```

---

## 损失记录与可视化

训练脚本会在 `--out` 目录下生成 `sdaa_loss.csv`：

```
step,loss
1,7.12
2,7.18
...
```

你可以使用 `matplotlib`/`pandas` 自行绘制，也可以直接用本文的 `compare_loss.py`。

---

## 日志对齐与误差评估

### 使用 `compare_loss.py`

```bash
python compare_loss.py \
  --sdaa ./outputs/test100/sdaa_loss.csv \
  --cuda ./logs/cuda_loss.csv \
  --align truncate \
  --smooth_window 5 \
  --smooth_poly 1 \
  --out_prefix cmp
```

生成内容：

* `cmp.txt`：打印 **MeanRelativeError / MeanAbsoluteError** 与 pass/fail 结论
* `cmp.csv`：对齐后的两列 loss
* `cmp.jpg`：平滑后的曲线对比图

核心评估规则示例：

```
MeanRelativeError: -0.00040
MeanAbsoluteError: -0.00196
Rule,mean_absolute_error -0.00196
pass mean_relative_error=-0.00040 <= 0.05 or mean_absolute_error=-0.00196 <= 0.0002
```

> 可根据实际需求调整阈值与对齐方式（`truncate`/`pad`/`stretch`）。

---

## 常见问题

1. **log.txt 为空**：脚本中使用 `print()` 而非 logger，可用 `tee` 重定向到文件。
2. **PrefetchLoader 报设备错误**：确保 `use_prefetcher=False`，并手动 `.to(device)`。
3. **分布式卡住**：检查 `SDAA_VISIBLE_DEVICES` 数量与 `--nproc_per_node` 是否一致，网络/通信库是否可用。
4. **显存不足**：调低 `--batch_size` 或增大 `--accum_steps`。


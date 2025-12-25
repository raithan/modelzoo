# U²-Net 模型在 SDAA 加速卡上的迁移与训练指南

> **目标**：在 **SDAA** 环境中，从零完成环境搭建、数据准备、模型训练（含 100 步快速验证）、损失记录与后续扩展。  
> **特点**：不依赖其它 GPU 生态；统一使用 `torch_sdaa` 后端；支持单卡 / 多卡、混合精度与对齐评测。

---

## 目录
1. [背景与思路](#背景与思路)
2. [软件与硬件要求](#软件与硬件要求)
3. [快速开始（极简 5 步）](#快速开始极简-5-步)
4. [环境搭建详解](#环境搭建详解)
5. [项目结构建议](#项目结构建议)
6. [数据准备（示例：COD10K_CAMO）](#数据准备示例cod10k_camo)
7. [训练脚本说明 (`train.py`)](#训练脚本说明-trainpy)
8. [运行 100 步快速验证](#运行-100-步快速验证)
9. [损失日志与对齐分析](#损失日志与对齐分析)
10. [多卡分布式训练](#多卡分布式训练)
11. [混合精度 (AMP)](#混合精度-amp)
12. [性能与调优建议](#性能与调优建议)
13. [常见问题 (FAQ)](#常见问题-faq)
14. [故障排查清单](#故障排查清单)
15. [附录：对比脚本片段](#附录对比脚本片段)

---

## 背景与思路
迁移的核心是：**替换默认计算设备 → 保障数据 / 模型均进入 SDAA → 验证功能正确性 → 才进行性能与精度对齐**。  
关键点：
- 使用 `import torch_sdaa` 注册后端；
- `torch.set_default_device("sdaa")` 统一放置新建张量；
- 训练脚本参数化（数据路径、步数、批次、损失日志）；
- 用 100 步快速验证数据通路、算子与损失曲线。

---

## 软件与硬件要求
| 组件 | 说明 |
|------|------|
| 驱动 / Runtime | SDAA 官方发布版本（示例：Driver/Runtime 2.1.x） |
| Python | 建议 3.9～3.11 |
| PyTorch | 与 `torch_sdaa` 对应版本（示例：2.4.x） |
| `torch_sdaa` | 与上面 PyTorch 精确匹配 |
| 依赖包 | `scikit-image`, `matplotlib`, `tqdm`, `numpy`, `scipy` |

---

## 快速开始（极简 5 步）

```bash
# 1. 创建环境
conda create -n u2net_sdaa python=3.10 -y
conda activate u2net_sdaa

# 2. 安装 PyTorch（CPU 版）+ torch_sdaa 匹配 wheel
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_sdaa-2.4.0-cp310-cp310-linux_x86_64.whl   # 路径按实际

# 3. 克隆项目
git clone https://github.com/xuebinqin/U-2-Net.git
cd U-2-Net

# 4. 安装常用依赖
pip install -U scikit-image matplotlib tqdm scipy

# 5. 运行 100 步快速验证（需准备数据，见后文）
SDAA_VISIBLE_DEVICES=0 \
python train.py --data_root /data/teco-data/COD10K_CAMO/TrainDataset \
  --img_dir Image --mask_dir GT --max_iter 100 --epochs 1 \
  --batch_size 1 --lr 1e-3 --model u2net --loss_log sdaa_loss.log

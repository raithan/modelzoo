# DeiT-Base on SDAA – README

> 仅依赖 PyTorch + Torch-SDAA 与 [timm]，无其它特殊框架改动。

---

## 1. 环境

```bash
conda create -n deit-sdaa python=3.10 -y
conda activate deit-sdaa

pip install timm==0.9.16 torchvision tqdm tensorboard pyyaml scipy matplotlib pandas
# 你的 torch 与 torch_sdaa 已安装即可
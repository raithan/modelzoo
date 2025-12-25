# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAM
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U‑2‑Net 训练脚本（SDAA / CUDA / CPU）+ loss 日志
"""
import os, glob, argparse, datetime
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# ---------- 设备选择 ----------
try:
    import torch_sdaa
    device_str = "sdaa" if torch.sdaa.is_available() else None
except (ImportError, AttributeError):
    device_str = None
if device_str is None:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device_str)
device = torch.device(device_str)
print(f"[INFO] Using device: {device}")

# ---------- 数据集 & 模型 ----------
from data_loader import SalObjDataset, RescaleT, RandomCrop, ToTensorLab
from model import U2NET, U2NETP

# ---------- 参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument("--data_root",  type=str, default="./train_data")
parser.add_argument("--img_dir",    type=str, default="Image")
parser.add_argument("--mask_dir",   type=str, default="GT")
parser.add_argument("--epochs",     type=int, default=1)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_iter",   type=int, default=100,
                    help="<=0 表示不限制步数")
parser.add_argument("--model",      type=str, default="u2net",
                    choices=["u2net", "u2netp"])
parser.add_argument("--loss_log",   type=str, default="sdaa_loss.log",
                    help="保存 loss 的文件名")
args = parser.parse_args()

# ---------- 文件列表 ----------
image_ext, label_ext = ".jpg", ".png"
img_glob   = os.path.join(args.data_root, args.img_dir, f"*{image_ext}")
train_imgs = sorted(glob.glob(img_glob))
mask_root  = os.path.join(args.data_root, args.mask_dir)
train_lbls = [os.path.join(mask_root,
               os.path.splitext(os.path.basename(p))[0] + label_ext)
               for p in train_imgs]
print(f"[INFO] train images: {len(train_imgs)}  train labels: {len(train_lbls)}")
if len(train_imgs) == 0:
    raise RuntimeError("未找到训练图像，请检查路径设置")

# ---------- DataLoader ----------
dataset = SalObjDataset(
    img_name_list=train_imgs,
    lbl_name_list=train_lbls,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)
    ])
)
loader = DataLoader(dataset, batch_size=args.batch_size,
                    shuffle=True, num_workers=2, pin_memory=False)

# ---------- 模型 / 优化器 / 损失 ----------
net = U2NET(3,1) if args.model=="u2net" else U2NETP(3,1)
net.to(device)
bce_loss = nn.BCELoss(reduction="mean")
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9,0.999))

def muti_bce(d0,d1,d2,d3,d4,d5,d6,gt):
    ls = [bce_loss(d,gt) for d in (d0,d1,d2,d3,d4,d5,d6)]
    return ls[0], sum(ls)

# ---------- 准备日志文件 ----------
log_path = args.loss_log
with open(log_path, "w") as f:
    f.write(f"# SDAA loss log | {datetime.datetime.now()}\n")
print(f"[INFO] loss will be logged to: {log_path}")

# ---------- 训练 ----------
ite, max_iter = 0, args.max_iter if args.max_iter>0 else float("inf")
for epoch in range(args.epochs):
    for data in loader:
        ite += 1
        imgs = data["image"].float().to(device)
        gts  = data["label"].float().to(device)

        optimizer.zero_grad()
        d0,d1,d2,d3,d4,d5,d6 = net(imgs)
        tar_loss, loss = muti_bce(d0,d1,d2,d3,d4,d5,d6,gts)
        loss.backward(); optimizer.step()

        # ---- 写日志 ----
        with open(log_path, "a") as f:
            f.write(f"{ite}\t{loss.item():.6f}\n")

        if ite % 10 == 0:
            print(f"[epoch {epoch+1}/{args.epochs} | step {ite}/{max_iter}] "
                  f"loss {loss.item():.4f}")

        if ite >= max_iter:
            print(f"[INFO] Reached max_iter={max_iter}, training stopped")
            break
    if ite >= max_iter:
        break

print("[INFO] Training finished ✔")
print(f"[INFO] Loss log saved at {os.path.abspath(log_path)}")

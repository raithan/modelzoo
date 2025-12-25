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
import os, argparse, contextlib, time
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------- device ----------
def has_sdaa():
    try:
        import torch_sdaa  # noqa: F401
        return True
    except Exception:
        return False

def get_device_amp(use_amp):
    if has_sdaa():
        dev = torch.device('sdaa:0')
        amp_ctx = (lambda: torch.autocast(device_type='sdaa', dtype=torch.float16)) if use_amp else contextlib.nullcontext
    elif torch.cuda.is_available():
        dev = torch.device('cuda:0')
        amp_ctx = torch.cuda.amp.autocast if use_amp else contextlib.nullcontext
    else:
        dev = torch.device('cpu')
        amp_ctx = contextlib.nullcontext
    return dev, amp_ctx

# ---------- dataset ----------
class BinMaskDataset(Dataset):
    def __init__(self, root, resize=512):
        self.root = Path(root)
        self.resize = resize
        self.to_tensor = transforms.ToTensor()
        imgd = self.root/'images'; maskd = self.root/'masks'
        exts = {'.jpg','.jpeg','.png'}
        self.pairs=[]
        for im in imgd.iterdir():
            if im.suffix.lower() in exts:
                m = maskd/im.stem
                for s in ['.png','.jpg','.jpeg']:
                    if (m.with_suffix(s)).exists():
                        self.pairs.append((im, m.with_suffix(s)))
                        break
        if not self.pairs:
            raise RuntimeError(f"No (image, mask) pairs found in {root}")
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        im, mk = self.pairs[i]
        I = Image.open(im).convert('RGB')
        M = Image.open(mk).convert('L')
        if self.resize:
            I = I.resize((self.resize,self.resize), Image.BILINEAR)
            M = M.resize((self.resize,self.resize), Image.NEAREST)
        I = self.to_tensor(I)
        M = (self.to_tensor(M)>0.5).float()
        return I, M

# ---------- model ----------
class DoubleConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c1,c2,3,1,1), nn.BatchNorm2d(c2), nn.ReLU(True),
            nn.Conv2d(c2,c2,3,1,1), nn.BatchNorm2d(c2), nn.ReLU(True),
        )
    def forward(self,x): return self.net(x)

class TinyUNet(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.d1=DoubleConv(3,ch);  self.p1=nn.MaxPool2d(2)
        self.d2=DoubleConv(ch,ch*2);self.p2=nn.MaxPool2d(2)
        self.d3=DoubleConv(ch*2,ch*4)
        self.u2=nn.ConvTranspose2d(ch*4,ch*2,2,2)
        self.d4=DoubleConv(ch*4,ch*2)
        self.u1=nn.ConvTranspose2d(ch*2,ch,2,2)
        self.d5=DoubleConv(ch*2,ch)
        self.out=nn.Conv2d(ch,1,1)
    def forward(self,x):
        c1=self.d1(x); p1=self.p1(c1)
        c2=self.d2(p1); p2=self.p2(c2)
        c3=self.d3(p2)
        u2=self.u2(c3); c4=self.d4(torch.cat([u2,c2],1))
        u1=self.u1(c4); c5=self.d5(torch.cat([u1,c1],1))
        return self.out(c5)   # logits

# ---------- loss ----------
class BCEDiceLoss(nn.Module):
    def forward(self, logits, target):
        import torch.nn.functional as F
        eps=1e-6
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probs = logits.sigmoid()
        inter = (probs*target).sum()
        dice = 1 - (2*inter+eps)/(probs.sum()+target.sum()+eps)
        return bce + dice

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    tot, n = 0.0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        tot += criterion(out,y).item() * x.size(0)
        n   += x.size(0)
    model.train()
    return tot / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_root','--train-root',dest='train_root',required=True)
    ap.add_argument('--val_root','--val-root',dest='val_root',required=True)
    ap.add_argument('--max_steps','--max-steps',dest='max_steps',type=int,default=100)
    ap.add_argument('--batch_size','--batch-size',dest='batch_size',type=int,default=8)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--log_file','--log-file',dest='log_file',default='sdaa_loss.log')
    ap.add_argument('--save_dir','--save-dir',dest='save_dir',default='weights')
    ap.add_argument('--load_ckpt','--load-ckpt',dest='load_ckpt',default=None)
    ap.add_argument('--model_ch','--model-ch',dest='model_ch',type=int,default=32)
    ap.add_argument('--resize',type=int,default=512)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device, amp_ctx = get_device_amp(args.amp)

    train_ds = BinMaskDataset(args.train_root, resize=args.resize)
    val_ds   = BinMaskDataset(args.val_root,   resize=args.resize)
    # 单进程，避免卡死
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    model = TinyUNet(ch=args.model_ch).to(device)

    # 可选加载权重
    if args.load_ckpt:
        p = Path(args.load_ckpt)
        if p.suffix == '.safetensors':
            from safetensors.torch import load_file
            sd = load_file(str(p))
        else:
            sd = torch.load(str(p), map_location='cpu', weights_only=True)
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"loaded: missing={len(miss)} unexpected={len(unexp)}")

    criterion = BCEDiceLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type=='cuda') else None

    with open(args.log_file, 'w') as f:
        f.write("# step,loss\n")

    print(f"len(train)={len(train_ds)} len(val)={len(val_ds)} device={device}", flush=True)

    best = 1e9
    step = 0
    model.train()
    while step < args.max_steps:
        for x,y in train_ld:
            if step >= args.max_steps: break
            x,y = x.to(device), y.to(device)
            with amp_ctx():
                out = model(x)
                loss= criterion(out,y)
            opt.zero_grad()
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            step += 1

            with open(args.log_file,'a') as f:
                f.write(f"{step},{loss.item():.6f}\n")

            if step % 20 == 0 or step == args.max_steps:
                v = validate(model, val_ld, criterion, device)
                print(f"[step {step}] train={loss.item():.4f} val={v:.4f}", flush=True)
                if v < best:
                    best = v
                    torch.save(model.state_dict(), Path(args.save_dir)/f"best_step{step}_{v:.4f}.pth")

    print("Done. Best val:", best)

if __name__ == "__main__":
    main()

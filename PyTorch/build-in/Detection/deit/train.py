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
# OF SUCH DAMAG
#!/usr/bin/env python3
# ==============================================================
# DeiT-Base training on SDAA (single / multi GPU)
# ==============================================================

import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.data import create_dataset, create_loader
from timm.models import create_model
from timm.utils import AverageMeter, setup_default_logging

# ------------------------ helpers ------------------------ #
def init_dist(backend='tccl'):
    dist.init_process_group(backend=backend)
    dist.barrier()

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(path, model, optimizer, lr_sched, epoch, dist_on=True):
    if dist_on and dist.get_rank() != 0:
        return
    state = {
        'model': (model.module.state_dict() if hasattr(model, 'module') else model.state_dict()),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_sched.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, path)
    print(f"==> Saved: {path}")

# ------------------------ args ------------------------ #
def parse_args():
    p = argparse.ArgumentParser()
    # data / model
    p.add_argument('--data', type=str, required=True, help='ImageNet root (with train/ & val/)')
    p.add_argument('--model', type=str, default='deit_base_patch16_224')
    p.add_argument('--num_classes', type=int, default=1000)
    p.add_argument('--pretrained_path', type=str, default='')

    # train
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--out', type=str, default='./outputs')
    p.add_argument('--num_workers', type=int, default=8)

    # debug / log
    p.add_argument('--max_steps', type=int, default=-1, help='stop after N global steps')
    p.add_argument('--loss_log', type=str, default='sdaa_loss.csv')

    # dist
    p.add_argument('--dist', action='store_true')
    p.add_argument('--backend', type=str, default='tccl')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

# ------------------------ main ------------------------ #
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    setup_default_logging(log_path=os.path.join(args.out, 'log.txt'))
    torch.manual_seed(args.seed)

    # distributed
    if args.dist:
        init_dist(args.backend)
        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank, local_rank = 0, 0

    device = torch.device('sdaa', local_rank)

    # dataset / loader
    dataset_train = create_dataset('imagenet', root=args.data, split='train', is_training=True, download=False)
    dataset_val   = create_dataset('imagenet', root=args.data, split='val',   is_training=False, download=False)

    loader_train = create_loader(
        dataset_train,
        input_size=(3, 224, 224),
        batch_size=args.batch_size,
        is_training=True,
        num_workers=args.num_workers,
        pin_memory=True,
        use_prefetcher=False,   # cuda-only prefetcher off
        distributed=args.dist
    )
    loader_val = create_loader(
        dataset_val,
        input_size=(3, 224, 224),
        batch_size=args.batch_size,
        is_training=False,
        num_workers=args.num_workers,
        pin_memory=True,
        use_prefetcher=False,
        distributed=False
    )

    # model
    model = create_model(args.model, pretrained=False, num_classes=args.num_classes)
    if args.pretrained_path and os.path.isfile(args.pretrained_path):
        print(f"==> Load pretrained from {args.pretrained_path}")
        ckpt = torch.load(args.pretrained_path, map_location='cpu')
        if 'model' in ckpt:
            ckpt = ckpt['model']
        msg = model.load_state_dict(ckpt, strict=False)
        print(msg)

    model.to(device)

    if args.dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss().to(device)

    # log
    global_step = 0
    log_f = None
    if rank == 0:
        log_f = open(os.path.join(args.out, args.loss_log), 'w', buffering=1)
        log_f.write("step,loss\n")

    # ---------------- loops ---------------- #
    def train_one_epoch(epoch):
        nonlocal global_step
        model.train()
        loss_meter = AverageMeter()
        optimizer.zero_grad()

        if args.dist and hasattr(loader_train.sampler, 'set_epoch'):
            loader_train.sampler.set_epoch(epoch)

        for step, (images, targets) in enumerate(loader_train):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type='sdaa', dtype=torch.bfloat16, enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            loss.backward()
            if (step + 1) % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_meter.update(loss.item(), images.size(0))
            if rank == 0 and log_f:
                log_f.write(f"{global_step+1},{loss.item():.6f}\n")

            global_step += 1
            if 0 < args.max_steps <= global_step:
                break

        lr_scheduler.step()
        if rank == 0:
            print(f"Epoch {epoch}: train_loss={loss_meter.avg:.4f}")

    @torch.no_grad()
    def validate(epoch):
        model.eval()
        top1_meter, top5_meter = AverageMeter(), AverageMeter()
        for images, targets in loader_val:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type='sdaa', dtype=torch.bfloat16, enabled=args.amp):
                outputs = model(images)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1_meter.update(acc1.item(), images.size(0))
            top5_meter.update(acc5.item(), images.size(0))
        if rank == 0:
            print(f"Epoch {epoch}: val@1={top1_meter.avg:.2f}  val@5={top5_meter.avg:.2f}")

    # main loop
    for epoch in range(args.epochs):
        train_one_epoch(epoch)
        validate(epoch)

        if (epoch + 1) % args.save_every == 0 or (0 < args.max_steps <= global_step):
            save_checkpoint(os.path.join(args.out, f'epoch{epoch+1:03d}.pth'),
                            model, optimizer, lr_scheduler, epoch, dist_on=args.dist)

        if 0 < args.max_steps <= global_step:
            break

    if log_f:
        log_f.close()

    if args.dist:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    main()

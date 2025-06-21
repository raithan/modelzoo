#BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
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
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
from torch_sdaa.utils import cuda_migrate  # 使用torch_sdaa自动迁移方法
from torch.sdaa import amp                  # 统一的 AMP 包装
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.sdaa 
import numpy as np

# ---------- DLLLogger 初始化 ----------
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
local_rank = int(os.environ.get("LOCAL_RANK", 0))
is_main = (local_rank == 0)
backends = [StdOutBackend(Verbosity.DEFAULT)]
# 只有 rank 0 负责写 JSON，避免多进程竞争
if is_main:
    backends.append(JSONStreamBackend(Verbosity.VERBOSE, "dlloger_example.json"))

json_logger = Logger(backends)

# ---- 定义元数据（只需一次，所有 rank 都执行即可）----
json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.ips",  {"unit": "imgs/s", "format": ":.3f",
                                    "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("val.loss",   {"unit": "", "GOAL": "MINIMIZE", "STAGE": "VAL"})
json_logger.metadata("val.ips",    {"unit": "imgs/s", "format": ":.3f",
                                    "GOAL": "MAXIMIZE", "STAGE": "VAL"})
# ------------------------------------


def broadcast_value(value, dtype=torch.float32):
    tensor = torch.tensor(value, dtype=dtype, device=torch.device(f"sdaa:{local_rank}"))
    dist.broadcast(tensor, src=0)
    return tensor.item() if dtype == torch.float32 else int(tensor.item())


def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    if dist.get_rank() == 0:
        seed = 20250620  # 固定种子
        random.seed(seed)
        lr = random.random()*0.0005 + 0.0000005
        augmentation_prob = random.random()*0.7
        # epoch = random.choice([100, 150, 200, 250])
        epoch = 100        
        decay_ratio = random.random()*0.8
        decay_epoch = int(epoch * decay_ratio)
    else:
        lr = augmentation_prob = decay_ratio = epoch = decay_epoch = 0  # 占位

    # 同步
    lr = broadcast_value(lr)
    augmentation_prob = broadcast_value(augmentation_prob)
    epoch = broadcast_value(epoch, dtype=torch.int32)
    decay_epoch = broadcast_value(decay_epoch, dtype=torch.int32)

    # 应用到 config
    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    if dist.get_rank() == 0:
        print(config)

        
    train_loader = get_loader(image_path=config.train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    test_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)

    # solver = Solver(config, train_loader, valid_loader, test_loader)
    solver = Solver(config, train_loader, valid_loader, test_loader, logger=json_logger)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--val_step', type=int, default=1)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='/data/teco-data/ISIC2018/train/')
    parser.add_argument('--valid_path', type=str, default='/data/teco-data/ISIC2018/valid/')
    parser.add_argument('--test_path', type=str, default='/data/teco-data/ISIC2018/test/')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    # ----- 1) IP/PORT & local_rank -----
    os.environ.setdefault("MASTER_ADDR", "localhost")     # IP
    os.environ.setdefault("MASTER_PORT", "12355")         # 任意空闲端口
    local_rank = int(os.environ.get("LOCAL_RANK", -1))    # torchrun 自动注入
    assert local_rank >= 0, "LOCAL_RANK 未设置，建议用 torchrun 启动"

    # ----- 2) 绑定当前 rank 的设备 -----
    device = torch.device(f"sdaa:{local_rank}")
    torch.sdaa.set_device(device)                         # tccl 前必须 set_device

    # ----- 3) 初始化进程组 -----
    dist.init_process_group(backend="tccl", init_method="env://")
    # =======================================================

    main(config)

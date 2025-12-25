# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its contributors
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
import random
from datetime import datetime

import albumentations as augs
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datasets import LandmarkDataset
from models import FAN
from train import train_model

if __name__ == '__main__':
    seed = 777
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    geometric_transform = augs.ShiftScaleRotate(scale_limit=0.2, rotate_limit=50, p=0.5,
                                                border_mode=cv2.BORDER_CONSTANT)

    content_transform = augs.Compose([augs.Blur(p=0.5),
                                      augs.ColorJitter(0.3, 0.3, 0.3, 0, 3, p=1.0),
                                      augs.ImageCompression(quality_lower=30, p=0.5),
                                      augs.CoarseDropout(min_holes=1, max_holes=8,
                                                         min_width=0.03125, min_height=0.03125,
                                                         max_width=0.125, max_height=0.125, p=0.5)])

    trainset_small = LandmarkDataset('./data/300w.tsv', LandmarkDataset.get_partitions('300w', 'train'),
                                     geometric_transform=geometric_transform, content_transform=content_transform,
                                     random_flip=True,
                                     config=LandmarkDataset.create_config(image_size=128, heatmap_size=32))
    valset_small = LandmarkDataset('./data/300w.tsv', LandmarkDataset.get_partitions('300w', 'val'),
                                   config=LandmarkDataset.create_config(image_size=128, heatmap_size=32))

    # Create data loaders
    trainset_loader = DataLoader(trainset_small, batch_size=32, shuffle=True, num_workers=32, pin_memory=True)
    valset_loader = DataLoader(valset_small, batch_size=16, shuffle=False, num_workers=32, pin_memory=True)
    print('Data loaders created.')

    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
        device = torch.device('cuda')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif torch.sdaa.is_available():
        print("SDAA is available. Training on GPU.")
        device = torch.device('sdaa')
        torch.sdaa.manual_seed(seed)
        torch.sdaa.manual_seed_all(seed)
    else:
        print("GPU is not available. Training on CPU.")
        device = torch.device('cpu')

    # Create the model
    if torch.cuda.is_available():
        fan = FAN.create_config(num_modules=2, use_avg_pool=False)
        fan = torch.nn.DataParallel(fan).to(device)
    else:
        # SDAA不支持DP
        fan = FAN.create_config(num_modules=2, use_avg_pool=False).to(device)

    print('Network initialised.')

    # Create the optimiser
    optimiser = torch.optim.RMSprop(fan.parameters(), lr=5e-4, momentum=0.9, weight_decay=0.0)
    print('Optimiser created.')

    # Create the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=8, gamma=0.7)
    print('LR scheduler created.')

    # Train the network
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with tqdm(desc='Epoch') as pbar1, tqdm(desc='Train batch') as pbar2, tqdm(desc='Validation batch') as pbar3:
        exp_01_01_ckpt_path = train_model(
            fan, optimiser, trainset_loader, valset_loader, 10, './logs', './checkpoints',
            f"exp_01_01_fan2_300w_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]}_utc",
            lr_scheduler=lr_scheduler, val_per_epoch=2, save_top_k=5,
            htm_mse_loss_weight=1.0, lmk_dist_loss_weight=0.01, rank_by='val_weighted_macc',
            train_weight_norm=trainset_loader.dataset.weight_norm,
            pbar_epochs=pbar1, pbar_train_batches=pbar2, pbar_val_batches=pbar3)
    del lr_scheduler
    del optimiser
    del fan
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Training finished, checkpoint saved to: {exp_01_01_ckpt_path}")

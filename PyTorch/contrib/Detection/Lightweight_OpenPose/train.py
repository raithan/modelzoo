# Adapted to tecorigin hardware 
import argparse
import cv2
import os
import time

import torch
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch_sdaa
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

from datasets.coco import CocoTrainDataset
from datasets.transformations import ConvertKeypoints, Scale, Rotate, CropPad, Flip
from modules.get_parameters import get_parameters_conv, get_parameters_bn, get_parameters_conv_depthwise
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.loss import l2_loss
from modules.load_state import load_state, load_from_mobilenet
from val import evaluate

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader

from tcap_dllogger import Logger, StdOutBackend,    JSONStreamBackend, Verbosity
json_logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE,    'dlloger_light.json'),
    ]
)

json_logger.metadata("train.loss", {"unit": "", "GOAL":    "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.ips",{"unit": "imgs/s",    "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})

import torch.nn as nn
def get_named_layers(model, layer_type):
    return [(name, m) for name, m in model.named_modules() if isinstance(m, layer_type)]

def build_optimizer_groups(model, base_lr):
    seen_params = set()
    param_groups = []

    def add_param_group(params, lr=None, weight_decay=None):
        group_params = []
        for p in params:
            if p.requires_grad and id(p) not in seen_params:
                group_params.append(p)
                seen_params.add(id(p))
        if group_params:
            group = {'params': group_params}
            if lr is not None:
                group['lr'] = lr
            if weight_decay is not None:
                group['weight_decay'] = weight_decay
            param_groups.append(group)

    # Helper functions
    def is_depthwise(m):
        return isinstance(m, nn.Conv2d) and m.groups == m.in_channels and m.in_channels == m.out_channels

    # ----- Main model -----
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if is_depthwise(m):
                add_param_group([m.weight], weight_decay=0)
            else:
                add_param_group([m.weight])
                if m.bias is not None:
                    add_param_group([m.bias], lr=base_lr * 2, weight_decay=0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                add_param_group([m.weight], weight_decay=0)
            if m.bias is not None:
                add_param_group([m.bias], lr=base_lr * 2, weight_decay=0)

    # ----- Module-specific lr scaling -----
    def add_module_params(submodule, w_lr=1.0, b_lr=2.0, base_wd=5e-4):
        for m in submodule.modules():
            if isinstance(m, nn.Conv2d):
                if is_depthwise(m):
                    add_param_group([m.weight], weight_decay=0)
                else:
                    add_param_group([m.weight], lr=base_lr * w_lr, weight_decay=base_wd)
                    if m.bias is not None:
                        add_param_group([m.bias], lr=base_lr * b_lr, weight_decay=0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    add_param_group([m.weight], lr=base_lr * w_lr, weight_decay=0)
                if m.bias is not None:
                    add_param_group([m.bias], lr=base_lr * b_lr, weight_decay=0)

    # Assign special lr to certain modules
    if hasattr(model, 'cpm'):
        add_module_params(model.cpm, w_lr=1.0, b_lr=2.0)
    if hasattr(model, 'initial_stage'):
        add_module_params(model.initial_stage, w_lr=1.0, b_lr=2.0)
    if hasattr(model, 'refinement_stages'):
        add_module_params(model.refinement_stages, w_lr=4.0, b_lr=8.0)

    return param_groups


def train(prepared_train_labels, train_images_folder, num_refinement_stages, base_lr, batch_size, batches_per_iter,
          num_workers, checkpoint_path, weights_only, from_mobilenet, checkpoints_folder, log_after,
          val_labels, val_images_folder, val_output_name, checkpoint_after, val_after):
    torch.sdaa.set_device(local_rank)
    device = torch.device(f'sdaa:{local_rank}')

    dist.init_process_group(backend='tccl', init_method='env://')

    net = PoseEstimationWithMobileNet(num_refinement_stages)
    net.to(device)

    stride = 8
    sigma = 7
    path_thickness = 1
    dataset = CocoTrainDataset(prepared_train_labels, train_images_folder,
                               stride, sigma, path_thickness,
                               transform=transforms.Compose([
                                   ConvertKeypoints(),
                                   Scale(),
                                   Rotate(pad=(128, 128, 128)),
                                   CropPad(pad=(128, 128, 128)),
                                   Flip()]))
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True)

    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    model = net.module if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net

    optimizer = torch.optim.Adam(
        build_optimizer_groups(model, base_lr),
        lr=base_lr,
        weight_decay=5e-4  # 通用设置，可被 param group 覆盖
    )

    scaler = GradScaler()

    num_iter = 0
    current_epoch = 0
    drop_after_epoch = [100, 200, 260]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.333)
    # 加载checkpoint（注意只由rank0加载，广播权重）
    if checkpoint_path and dist.get_rank() == 0:
        if num_iter >= 150:
            sys.exit(0)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if from_mobilenet:
            load_from_mobilenet(net.module, checkpoint)
        else:
            load_state(net.module, checkpoint)
            if not weights_only:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                num_iter = checkpoint['iter']
                current_epoch = checkpoint['current_epoch']

    dist.barrier()  # 等待所有进程加载完权

    # net = DataParallel(net).cuda()

    net.train()
    start_time = time.time()
    for epochId in range(current_epoch, 1):
        train_sampler.set_epoch(epochId)  # 每轮打乱数据保证随机性
        # scheduler.step()
        total_losses = [0, 0] * (num_refinement_stages + 1)  # heatmaps loss, paf loss per stage
        batch_per_iter_idx = 0
        for batch_data in train_loader:
            if batch_per_iter_idx == 0:
                optimizer.zero_grad()

            images = batch_data['image'].cuda()
            keypoint_masks = batch_data['keypoint_mask'].cuda()
            paf_masks = batch_data['paf_mask'].cuda()
            keypoint_maps = batch_data['keypoint_maps'].cuda()
            paf_maps = batch_data['paf_maps'].cuda()

            # stages_output = net(images)

            # losses = []
            # for loss_idx in range(len(total_losses) // 2):
            #     losses.append(l2_loss(stages_output[loss_idx * 2], keypoint_maps, keypoint_masks, images.shape[0]))
            #     losses.append(l2_loss(stages_output[loss_idx * 2 + 1], paf_maps, paf_masks, images.shape[0]))
            #     total_losses[loss_idx * 2] += losses[-2].item() / batches_per_iter
            #     total_losses[loss_idx * 2 + 1] += losses[-1].item() / batches_per_iter

            # loss = losses[0]

            with autocast():
                stages_output = net(images)
                losses = []
                for loss_idx in range(len(total_losses) // 2):
                    losses.append(l2_loss(stages_output[loss_idx * 2], keypoint_maps, keypoint_masks, images.shape[0]))
                    losses.append(l2_loss(stages_output[loss_idx * 2 + 1], paf_maps, paf_masks, images.shape[0]))
                    total_losses[loss_idx * 2] += losses[-2].item() / batches_per_iter
                    total_losses[loss_idx * 2 + 1] += losses[-1].item() / batches_per_iter
                loss = sum(losses) / batches_per_iter


            # for loss_idx in range(1, len(losses)):
            #     loss += losses[loss_idx]
            # loss /= batches_per_iter
            # loss.backward()
            scaler.scale(loss).backward()            
            batch_per_iter_idx += 1
            if batch_per_iter_idx == batches_per_iter:
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()

                batch_per_iter_idx = 0
                num_iter += 1
            else:
                continue

            if num_iter >= 300:
                print(f"Reached {num_iter} iterations, stopping training.")
                break

            if num_iter % log_after == 0:
                elapsed_time = time.time() - start_time
                ips = (batch_size * log_after) / elapsed_time
                json_logger.log(
                step = (epochId, num_iter),
                data = {
                        "rank":os.environ["LOCAL_RANK"],
                        "train.loss":total_losses[2] / log_after, 
                        # "train.loss":loss,
                        "train.ips":ips * log_after,
                        },
                verbosity=Verbosity.DEFAULT,
                )
                print('Iter: {}'.format(num_iter))
                for loss_idx in range(len(total_losses) // 2):
                    print('\n'.join(['stage{}_pafs_loss:     {}', 'stage{}_heatmaps_loss: {}']).format(
                        loss_idx + 1, total_losses[loss_idx * 2 + 1] / log_after,
                        loss_idx + 1, total_losses[loss_idx * 2] / log_after))
                for loss_idx in range(len(total_losses)):
                    total_losses[loss_idx] = 0
            if num_iter % checkpoint_after == 0:
                snapshot_name = '{}/checkpoint_iter_{}.pth'.format(checkpoints_folder, num_iter)
                torch.save({'state_dict': net.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iter': num_iter,
                            'current_epoch': epochId},
                           snapshot_name)
            if num_iter % val_after == 0:
                print('Validation...')
                evaluate(val_labels, val_output_name, val_images_folder, net)
                net.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-train-labels', type=str, required=True,
                        help='path to the file with prepared annotations')
    parser.add_argument('--train-images-folder', type=str, required=True, help='path to COCO train images folder')
    parser.add_argument('--num-refinement-stages', type=int, default=1, help='number of refinement stages')
    parser.add_argument('--base-lr', type=float, default=4e-5, help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--batches-per-iter', type=int, default=1, help='number of batches to accumulate gradient from')
    parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint to continue training from')
    parser.add_argument('--from-mobilenet', action='store_true',
                        help='load weights from mobilenet feature extractor')
    parser.add_argument('--weights-only', action='store_true',
                        help='just initialize layers with pre-trained weights and start training from the beginning')
    parser.add_argument('--experiment-name', type=str, default='default',
                        help='experiment name to create folder for checkpoints')
    parser.add_argument('--log-after', type=int, default=1, help='number of iterations to print train loss')

    parser.add_argument('--val-labels', type=str, required=True, help='path to json with keypoints val labels')
    parser.add_argument('--val-images-folder', type=str, required=True, help='path to COCO val images folder')
    parser.add_argument('--val-output-name', type=str, default='detections.json',
                        help='name of output json file with detected keypoints')
    parser.add_argument('--checkpoint-after', type=int, default=5000,
                        help='number of iterations to save checkpoint')
    parser.add_argument('--val-after', type=int, default=5000,
                        help='number of iterations to run validation')
    args = parser.parse_args()

    # 从环境变量读取分布式相关信息
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ.get('WORLD_SIZE', 1))


    checkpoints_folder = '{}_checkpoints'.format(args.experiment_name)
    # if not os.path.exists(checkpoints_folder):
    #     os.makedirs(checkpoints_folder)
    if local_rank == 0 and not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    train(args.prepared_train_labels, args.train_images_folder, args.num_refinement_stages, args.base_lr, args.batch_size,
          args.batches_per_iter, args.num_workers, args.checkpoint_path, args.weights_only, args.from_mobilenet,
          checkpoints_folder, args.log_after, args.val_labels, args.val_images_folder, args.val_output_name,
          args.checkpoint_after, args.val_after)

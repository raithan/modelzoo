from __future__ import print_function
from torch_sdaa.utils import cuda_migrate
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import AnnotationTransform, VOCDetection, detection_collate, preproc, cfg
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.faceboxes import FaceBoxes

parser = argparse.ArgumentParser(description='FaceBoxes Training')
parser.add_argument('--training_dataset', default='./data/WIDER_FACE', help='Training dataset directory')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=300, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

img_dim = 1024 # only 1024 is supported
rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
num_gpu = args.ngpu
num_workers = args.num_workers
batch_size = args.batch_size
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
max_epoch = args.max_epoch
training_dataset = args.training_dataset
save_folder = args.save_folder
gpu_train = cfg['gpu_train']

net = FaceBoxes('train', img_dim, num_classes)
print("Printing net...")
print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net, device_ids=list(range(num_gpu)))
else:
    net = net.to('sdaa')

device = torch.device('cuda:0' if gpu_train else 'cpu')
cudnn.benchmark = True
net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to(device)


def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = VOCDetection(training_dataset, preproc(img_dim, rgb_mean), AnnotationTransform())

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (200 * epoch_size, 250 * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), save_folder + 'FaceBoxes_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.to(device)
        targets = [anno.to(device) for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || L: {:.4f} C: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'.format(epoch, max_epoch, (iteration % epoch_size) + 1, epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), save_folder + 'Final_FaceBoxes.pth')


# def register_debug_hook(module):
#     def forward_hook(m, input, output):
#         print(f"Forward: {m} | Input shape: {input[0].shape}, Output shape: {output.shape}")
#     module.register_forward_hook(forward_hook)

#     def backward_hook(m, grad_input, grad_output):
#         print(f"Backward: {m} | Grad output shape: {grad_output[0].shape}")
#     module.register_full_backward_hook(backward_hook)

# for m in net.modules():
#     if isinstance(m, torch.nn.Conv2d):
#         register_debug_hook(m)
        

# def train():
#     net.train()
#     epoch = 0 + args.resume_epoch
#     print('Loading Dataset...')

#     dataset = VOCDetection(training_dataset, preproc(img_dim, rgb_mean), AnnotationTransform())

#     epoch_size = math.ceil(len(dataset) / batch_size)
#     max_iter = max_epoch * epoch_size

#     stepvalues = (200 * epoch_size, 250 * epoch_size)
#     step_index = 0

#     if args.resume_epoch > 0:
#         start_iter = args.resume_epoch * epoch_size
#     else:
#         start_iter = 0

#     # for param in net.conv1.parameters():
#     #     param.requires_grad = False
#     # print("Frozen layer: conv1")

#     # ======【新增】打印模型结构摘要======
#     print("Model Structure Summary:")
#     print(net)

#     for iteration in range(start_iter, max_iter):
#         if iteration % epoch_size == 0:
#             # create batch iterator
#             batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
#             if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
#                 torch.save(net.state_dict(), save_folder + 'FaceBoxes_epoch_' + str(epoch) + '.pth')
#             epoch += 1

#         load_t0 = time.time()
#         if iteration in stepvalues:
#             step_index += 1
#         lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

#         # load train data
#         try:
#             images, targets = next(batch_iterator)
#         except StopIteration:
#             print(f"[ERROR] BatchIterator exhausted at iteration {iteration}, restarting...")
#             batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
#             images, targets = next(batch_iterator)

#         # ======【新增】检查输入数据有效性======
#         if torch.isnan(images).any() or torch.isinf(images).any():
#             print(f"[WARNING] NaN/Inf detected in input images at iteration {iteration}")
#         if any(torch.isnan(t).any() or torch.isinf(t).any() for t in targets):
#             print(f"[WARNING] NaN/Inf detected in targets at iteration {iteration}")

#         images = images.to(device)
#         targets = [anno.to(device) for anno in targets]

#         # ======【新增】打印输入图像 shape======
#         print(f"Iteration {iteration}: Input image shape: {images.shape}")

#         # forward
#         try:
#             out = net(images)
#         except Exception as e:
#             print(f"[ERROR] Forward failed at iteration {iteration}")
#             print(f"Input tensor shape: {images.shape}")
#             print(f"Exception message: {e}")
#             raise

#         # ======【新增】打印网络输出形状======
#         if isinstance(out, (list, tuple)):
#             for i, o in enumerate(out):
#                 print(f"Output[{i}] shape: {o.shape}")
#         else:
#             print(f"Output shape: {out.shape}")

#         # backprop
#         optimizer.zero_grad()
#         try:
#             loss_l, loss_c = criterion(out, priors, targets)
#             loss = cfg['loc_weight'] * loss_l + loss_c
#         except Exception as e:
#             print(f"[ERROR] Loss calculation failed at iteration {iteration}")
#             print(f"Output shapes: {[o.shape for o in out]}")
#             print(f"Priors shape: {priors.shape}")
#             print(f"Targets shapes: {[t.shape for t in targets]}")
#             print(f"Exception message: {e}")
#             raise

#         # ======【新增】检查 loss 是否为 NaN======
#         if torch.isnan(loss):
#             print(f"[ERROR] NaN detected in loss at iteration {iteration}")
#             print(f"Loss_l: {loss_l.item()}, Loss_c: {loss_c.item()}")
#             # 可以在这里保存当前 batch 和 model 状态用于调试
#             torch.save({
#                 'iteration': iteration,
#                 'images': images,
#                 'targets': targets,
#                 'outputs': out,
#                 'model_state': net.state_dict(),
#             }, f"debug_checkpoint_iter_{iteration}.pth")
#             exit()

#         # ======【新增】梯度监控======
#         def print_grad(name, module):
#             def hook(grad):
#                 if torch.isnan(grad).any():
#                     print(f"[WARNING] NaN in gradients of {name}")
#                 if torch.isinf(grad).any():
#                     print(f"[WARNING] Inf in gradients of {name}")
#             return hook

#         for name, param in net.named_parameters():
#             if param.requires_grad and param.grad_fn is not None:
#                 param.register_hook(print_grad(name, param))

#         # 执行反向传播
#         try:
#             loss.backward()
#         except Exception as e:
#             print(f"[ERROR] Backward failed at iteration {iteration}")
#             print(f"Loss value: {loss.item()}")
#             print(f"Exception message: {e}")
#             # 保存当前状态以便后续分析
#             torch.save({
#                 'iteration': iteration,
#                 'images': images,
#                 'targets': targets,
#                 'outputs': out,
#                 'loss': loss,
#                 'model_state': net.state_dict(),
#             }, f"backward_failure_iter_{iteration}.pth")
#             raise

#         optimizer.step()
#         load_t1 = time.time()
#         batch_time = load_t1 - load_t0
#         eta = int(batch_time * (max_iter - iteration))
#         print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || L: {:.4f} C: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'.format(epoch, max_epoch, (iteration % epoch_size) + 1, epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

#     torch.save(net.state_dict(), save_folder + 'Final_FaceBoxes.pth')



def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()

from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import torch.distributed as dist

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from reid import datasets
from reid import models
from reid.trainers_partloss import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
    
    device = torch.device("cuda")
    seed = 777
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU


def get_data(name, data_dir, height, width, batch_size, workers, is_distributed=False):
    root = osp.join(data_dir, name)
    root = data_dir
    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_train_ids

    train_transformer = T.Compose([
        T.RectScale(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    # 训练集使用DistributedSampler进行数据分片
    train_sampler = DistributedSampler(dataset.train) if is_distributed else None
    
    train_loader = DataLoader(
        Preprocessor(dataset.train, root=osp.join(dataset.images_dir,dataset.train_path),
                    transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=(train_sampler is None), pin_memory=True, drop_last=True,
        sampler=train_sampler)

    query_loader = DataLoader(
        Preprocessor(dataset.query, root=osp.join(dataset.images_dir,dataset.query_path),
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=osp.join(dataset.images_dir,dataset.gallery_path),
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, query_loader, gallery_loader, train_sampler

def main(args):
    # 初始化分布式环境
    if args.distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # 只有主进程(通常rank=0)才重定向输出到日志文件
    if not args.evaluate and (not args.distributed or args.rank == 0):
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # 创建数据加载器
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)
    dataset, num_classes, train_loader, query_loader, gallery_loader, train_sampler = \
        get_data(args.dataset, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, args.distributed)

    # 创建模型
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes,
                          cut_at_pooling=False, FCN=True)
    
    # 将模型移至当前GPU
    model = model.to(device)

    # 加载检查点
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        
        # 调整状态字典加载方式
        state_dict = checkpoint['state_dict']
        # 如果是从DP/DDP模型加载，需要移除module.前缀
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        if not args.distributed or args.rank == 0:
            print("=> Start epoch {}  best top1 {:.1%}"
                  .format(start_epoch, best_top1))

    # 包装模型为DDP
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 转换BatchNorm为SyncBatchNorm
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], 
                                                   output_device=args.local_rank, find_unused_parameters=True)

    # 评估器
    evaluator = Evaluator(model)
    if args.evaluate:
        if not args.distributed or args.rank == 0:
            print("Test:")
        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)
        if args.distributed:
            dist.destroy_process_group()  # 销毁进程组
        return

    # 损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # 优化器
    if hasattr(model if not args.distributed else model.module, 'base'):
        base_model = model if not args.distributed else model.module
        base_param_ids = set(map(id, base_model.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': base_model.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # 训练器
    trainer = Trainer(model, criterion, 0, 0, SMLoss_mode=0)

    # 学习率调度
    def adjust_lr(epoch):
        step_size = 60 if args.arch == 'inception' else args.step_size
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # 开始训练
    for epoch in range(start_epoch, args.epochs):
        # 如果使用DistributedSampler，需要设置epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer ,args.max_steps)
        
        # 只在主进程保存模型
        if not args.distributed or args.rank == 0:
            is_best = True
            save_checkpoint({
                'state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

    # 最终测试
    if not args.distributed or args.rank == 0:
        print('Test with best model:')
        checkpoint = load_checkpoint(osp.join(args.logs_dir, 'checkpoint.pth.tar'))
        model_to_load = model.module if args.distributed else model
        model_to_load.load_state_dict(checkpoint['state_dict'])
        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)
    
    # 销毁进程组
    if args.distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # 数据参数
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    # 模型参数
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    # 优化器参数
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # 训练配置
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--step-size',type=int, default=40)
    parser.add_argument('--max-steps', type=int, default=None,
                        help="Maximum number of training steps. 0 means no limit.")
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--print-freq', type=int, default=1)
    # 分布式训练参数
    parser.add_argument('--distributed', action='store_true',
                        help="Use distributed training")
    # 其他参数
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    args = parser.parse_args()
    
    # 创建日志目录
    if not osp.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    
    main(args)

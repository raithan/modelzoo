import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv.runner import init_dist, build_optimizer, IterBasedRunner, IterTimerHook
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def move_data_to_device(data, device):
    """Recursively move data to the specified device"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_data_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_data_to_device(item, device) for item in data)
    else:
        return data


def unpack_data_batch(data_batch, device):
    """Unpack DataContainer objects in data batch and move to device"""
    if isinstance(data_batch, dict):
        unpacked = {}
        for key, value in data_batch.items():
            if hasattr(value, 'data'):
                # This is a DataContainer, unpack its data
                if isinstance(value.data, list) and len(value.data) > 0:
                    unpacked[key] = move_data_to_device(value.data[0], device)
                else:
                    unpacked[key] = move_data_to_device(value.data, device)
            else:
                unpacked[key] = move_data_to_device(value, device)
        return unpacked
    elif isinstance(data_batch, (list, tuple)):
        return [unpack_data_batch(item, device) for item in data_batch]
    else:
        return move_data_to_device(data_batch, device)


def wrap_model_for_sda(model, device):
    """Wrap model to handle DataContainer and SDA device properly"""
    original_train_step = model.train_step
    
    def custom_train_step(data_batch, optimizer, **kwargs):
        # Unpack DataContainer objects and move to device
        unpacked_data = unpack_data_batch(data_batch, device)
        # Call original train step
        return original_train_step(unpacked_data, optimizer, **kwargs)
    
    model.train_step = custom_train_step
    return model


def init_sda_distributed():
    """Initialize a single-process distributed environment for SDA"""
    try:
        # Check if distributed is already initialized
        if torch.distributed.is_available() and not torch.distributed.is_initialized():
            # Set environment variables
            os.environ.setdefault('RANK', '0')
            os.environ.setdefault('WORLD_SIZE', '1')
            os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
            os.environ.setdefault('MASTER_PORT', '12355')
            
            # Initialize process group
            torch.distributed.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='env://',
                world_size=1,
                rank=0
            )
            print("Initialized single-process distributed environment for SDAA")
    except Exception as e:
        print(f"Warning: Could not initialize distributed environment: {e}")


def train_model_iterson(model, dataset, cfg, logger, device):
    """Custom training loop without DataParallel"""
    
    # Build dataloader
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            num_gpus=1,  # Force single GPU
            dist=False,
            seed=cfg.seed,
            # Disable pin_memory for SDA compatibility
            pin_memory=False,
            # Disable persistent workers
            persistent_workers=False) for ds in dataset
    ]
    
    # Build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)
    
    # Build runner
    runner = IterBasedRunner(
        model=model,
        batch_processor=None,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=dict()
    )
    
    # Register hooks
    runner.register_training_hooks(
        lr_config=cfg.lr_config,
        optimizer_config=cfg.get('optimizer_config', dict()),
        checkpoint_config=cfg.checkpoint_config,
        log_config=cfg.log_config,
    )
    
    runner.register_hook(IterTimerHook())
    
    # Load checkpoint if specified
    if cfg.get('resume_from', None):
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from', None):
        runner.load_checkpoint(cfg.load_from)
    
    # Get max iterations
    max_iters = cfg.runner.get('max_iters', 160000)
    
    # Run training
    runner.run(data_loaders, cfg.workflow, max_iters)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    # Force single GPU training
    cfg.gpu_ids = [0]
    
    # Disable workers for SDA compatibility
    if hasattr(cfg.data, 'workers_per_gpu'):
        cfg.data.workers_per_gpu = 0

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Custom single GPU training')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # Initialize SDA distributed environment
    init_sda_distributed()

    # Build model
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    # Move model to SDA device
    device = torch.device('sdaa:0')
    model = model.to(device)
    logger.info(model)
    
    # Wrap model to handle DataContainer properly and ensure device consistency
    model = wrap_model_for_sda(model, device)

    # Build datasets
    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    
    # Custom training without DataParallel
    train_model_iterson(model, datasets, cfg, logger, device)


if __name__ == '__main__':
    main()
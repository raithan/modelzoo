import argparse
import os
import os.path as osp
from copy import deepcopy

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION


def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, else auto resume from latest checkpoint')
    parser.add_argument(
        '--amp',
        default='True',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to actual batch size')
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='disable pin_memory in dataloaders')
    parser.add_argument(
        '--no-persistent-workers',
        action='store_true',
        help='disable persistent_workers in dataloaders')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some config settings, in xxx=yyy format')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args
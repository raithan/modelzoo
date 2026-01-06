import argparse
from pathlib import Path

# 脚本路径：run_scripts/argument.py，工作目录为上级目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # 工作目录：/data/bigc-data/ltb/finetune_segment_anything_tutorial


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # 权重路径：工作目录下的weights文件夹
    parser.add_argument(
        '--sam-weights', '--w', 
        type=str, 
        default=ROOT / 'weights/sam_vit_b_01ec64.pth', 
        help='original sam weights path'
    )
    parser.add_argument(
        '--model-type', '--type', 
        type=str, 
        default='vit_b', 
        help='sam model type: vit_b, vit_l, vit_h'
    )
    # 数据集路径：工作目录下的data_example/VOCdevkit
    parser.add_argument(
        '--data', 
        type=str, 
        default=ROOT / 'data_example/VOCdevkit', 
        help='your VOCdevkit dataset path'
    )
    parser.add_argument(
        '--point-prompt', 
        type=bool, 
        default=True, 
        help='use point prompt'
    )
    parser.add_argument(
        '--box-prompt', 
        type=bool, 
        default=True, 
        help='use box prompt'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=1, 
        help='total training epochs'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=1, 
        help='total batch size for all GPUs, must be 1 for voc'
    )
    # 保存路径：工作目录下的runs文件夹
    parser.add_argument(
        '--save_dir', 
        default=ROOT / 'runs', 
        help='path to save checkpoint'
    )
    parser.add_argument(
        '--device', 
        default='0', 
        help='cuda device only one, 0 or 1 or 2...'
    )
    return parser.parse_known_args()[0] if known else parser.parse_args()
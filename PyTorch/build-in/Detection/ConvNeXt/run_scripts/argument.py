def parse_args():
    """Parse command-line arguments for ConvNeXt+SDAA quick experiments."""
    parser = argparse.ArgumentParser(description='Quick train / debug')

    # --- 基本 ---
    parser.add_argument('--config', help='MMEngine / MMDet style config path')
    parser.add_argument('--work-dir', default='./work_dirs',
                        help='Directory to save logs and checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from given checkpoint (or "auto")')

    # --- 数据与训练 ---
    parser.add_argument('--data-path', default='data/teco-data/imagenet',
                        help='ImageNet root dir with train/ val/ subfolders')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='sdaa',
                        help='sdaa | cuda | cpu')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Stop after N iters (debug)')

    # --- 训练选项 ---
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed-precision training')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip val during training')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='Override config keys, e.g. model.backbone.depth=12')

    # --- dataloader 性能调节 ---
    parser.add_argument('--pin-mem', action='store_true',
                        help='Use pinned-memory dataloader')
    parser.add_argument('--persistent-workers', action='store_true',
                        help='Use persistent dataloader workers')

    # ========== 以下暂不用分布式，可按需再开启 ==========
    # parser.add_argument('--launcher', ... )
    # parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--nnodes', type=int, default=1)
    # parser.add_argument('--nproc-per-node', type=int, default=1)
    # parser.add_argument('--master-addr', default='localhost')
    # parser.add_argument('--master-port', default='29500')
    # parser.add_argument('--node-rank', type=int, default=0)
    # ====================================================

    args = parser.parse_args()

    # 如果需要自动设置 LOCAL_RANK（单进程即可）
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args
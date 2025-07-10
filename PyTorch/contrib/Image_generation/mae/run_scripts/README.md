## 参数介绍

| 参数名               | 说明                                                                      | 示例                              |
| ----------------- | ----------------------------------------------------------------------- | ------------------------------- |
| `--batch_size`    | 每个 GPU 上的训练 batch 大小，**总 batch 大小 = batch\_size × accum\_iter × GPU 数** | `--batch_size 64`               |
| `--epochs`        | 总训练 epoch 数                                                             | `--epochs 400`                  |
| `--accum_iter`    | 梯度累计的迭代次数，用于显存不足时模拟更大的 batch size                                       | `--accum_iter 2`                |
| `--model`         | 模型名称（如 MAE 使用的 ViT 结构）                                                  | `--model mae_vit_large_patch16` |
| `--input_size`    | 输入图像的尺寸（长宽为 `input_size × input_size`）                                  | `--input_size 224`              |
| `--mask_ratio`    | 掩码比率，用于 MAE 的自监督遮挡训练                                                    | `--mask_ratio 0.75`             |
| `--norm_pix_loss` | 是否使用归一化像素作为 loss 计算目标                                                   | `--norm_pix_loss`               |
| `--weight_decay`  | 权重衰减（L2 正则项）                                                            | `--weight_decay 0.05`           |
| `--lr`            | 学习率（可为空，默认根据 `blr` 动态计算）                                                | `--lr 0.0001`                   |
| `--blr`           | 基础学习率，实际学习率 = base\_lr × total\_batch\_size / 256                       | `--blr 1e-3`                    |
| `--min_lr`        | 最小学习率（用于 cosine/lambda 等调度器）                                            | `--min_lr 1e-5`                 |
| `--warmup_epochs` | 学习率 warmup 的 epoch 数                                                    | `--warmup_epochs 40`            |
| `--data_path`     | 训练数据集的路径                                                                | `--data_path /path/to/imagenet` |
| `--output_dir`    | 模型输出保存目录                                                                | `--output_dir ./output_dir`     |
| `--log_dir`       | TensorBoard 日志输出目录                                                      | `--log_dir ./output_dir`        |
| `--device`        | 训练使用的设备（如 `cuda`、`cpu`）                                                 | `--device cuda`                 |
| `--seed`          | 随机数种子，确保实验可复现                                                           | `--seed 42`                     |
| `--resume`        | 继续训练时加载的 checkpoint 文件路径                                                | `--resume ./checkpoint.pth`     |
| `--start_epoch`   | 起始 epoch（通常在 resume 时指定）                                                | `--start_epoch 10`              |
| `--num_workers`   | DataLoader 使用的线程数量                                                      | `--num_workers 10`              |
| `--pin_mem`       | 是否启用 DataLoader 的内存锁页（pin memory）以加速 CPU → GPU 传输                       | `--pin_mem`（启用）                 |
| `--no_pin_mem`    | 关闭 pin memory                                                           | `--no_pin_mem`                  |
| `--world_size`    | 分布式训练的总进程数                                                              | `--world_size 4`                |
| `--local_rank`    | 当前进程在分布式训练中的 rank ID（通常由启动脚本自动指定）                                       | `--local_rank 0`                |
| `--dist_on_itp`   | 是否在 ITP 平台上启用分布式训练（特定平台标志）                                              | `--dist_on_itp`                 |
| `--dist_url`      | 分布式训练初始化使用的地址（如环境变量或 `tcp://`）                                          | `--dist_url env://`             |


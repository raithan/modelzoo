## 参数介绍
| 参数名                       | 说明                                             | 示例                                                                                    |
| ------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------- |
| `--prepared-train-labels` | 已预处理的 COCO 关键点注释（JSON）路径                       | `--prepared-train-labels prepared_train_annotation.pkl` |
| `--train-images-folder`   | COCO 训练图像文件夹路径                                 | `--train-images-folder  /data/teco-data/COCO/images/train2017/`                                              |
| `--num-refinement-stages` | 网络中的 refinement stage 数量（即多个阶段预测，逐步精细化）        | `--num-refinement-stages 2`                                                           |
| `--base-lr`               | 初始学习率                                          | `--base-lr 4e-5`                                                                      |
| `--batch-size`            | 每个 batch 的样本数                                  | `--batch-size 16`                                                                     |
| `--batches-per-iter`      | 每次优化器 step 前累积多少个 batch 的梯度（梯度累积）              | `--batches-per-iter 2`                                                                |
| `--num-workers`           | 数据加载时使用的子进程数量（即 DataLoader 的 `num_workers` 参数） | `--num-workers 8`                                                                     |
| `--checkpoint-path`       | 用于继续训练的 checkpoint 权重文件路径                      | `--checkpoint-path ./mobilenet_sgd_68.848.pth.tar`                           |
| `--from-mobilenet`        | 指定是否从 MobileNet 的预训练模型加载特征提取器权重                | `--from-mobilenet`（不加参数值，设置该 flag 即启用）                                                |
| `--weights-only`          | 只加载权重，不加载优化器与调度器状态，从头开始训练                      | `--weights-only`                                                                      |
| `--experiment-name`       | 当前实验的名称（用于保存 checkpoint 文件夹）                   | `--experiment-name pose_coco_exp1`                                                    |
| `--log-after`             | 每多少次迭代打印一次训练损失和吞吐率                             | `--log-after 10`                                                                      |
| `--val-labels`            | 验证集关键点标签（COCO 验证 JSON 格式）路径                    | `--val-labels val_subset.json`                       |
| `--val-images-folder`     | COCO 验证图像文件夹路径                                 | `--val-images-folder /data/teco-data/COCO/images/val2017/`                                                  |
| `--val-output-name`       | 模型在验证集推理后输出的 JSON 文件名                          | `--val-output-name detections.json`                                                   |
| `--checkpoint-after`      | 每多少次迭代保存一次 checkpoint                          | `--checkpoint-after 5000`                                                             |
| `--val-after`             | 每多少次迭代在验证集上运行一次评估                              | `--val-after 5000`                                                                    |


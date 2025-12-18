## 参数介绍
| 参数名 | 说明 | 示例 |
|--------|------|------|
| `--image_size` | 输入图像的尺寸（高度和宽度） | `--image_size 224` |
| `--t` | R2U_Net或R2AttU_Net中递归步骤的重复次数 | `--t 3` |
| `--img_ch` | 输入图像的通道数 | `--img_ch 3` |
| `--output_ch` | 输出图像的通道数 | `--output_ch 1` |
| `--num_epochs` | 总训练epoch数 | `--num_epochs 100` |
| `--num_epochs_decay` | 学习率衰减开始的epoch | `--num_epochs_decay 70` |
| `--batch_size` | 每个batch的样本数 | `--batch_size 1` |
| `--num_workers` | 数据加载的线程数 | `--num_workers 2` |
| `--lr` | 初始学习率 | `--lr 0.0002` |
| `--beta1` | Adam优化器的第一个动量参数 | `--beta1 0.5` |
| `--beta2` | Adam优化器的第二个动量参数 | `--beta2 0.999` |
| `--augmentation_prob` | 数据增强的概率 | `--augmentation_prob 0.4` |
| `--log_step` | 每隔多少step记录一次日志 | `--log_step 1` |
| `--val_step` | 每隔多少epoch验证一次 | `--val_step 1` |
| `--mode` | 运行模式（训练/测试等） | `--mode train` |
| `--model_type` | 模型类型（U_Net/R2U_Net等） | `--model_type U_Net` |
| `--model_path` | 模型保存路径 | `--model_path ./models` |
| `--train_path` | 训练数据路径 | `--train_path /data/teco-data/ISIC2018/train/` |
| `--valid_path` | 验证数据路径 | `--valid_path /data/teco-data/ISIC2018/valid/` |
| `--test_path` | 测试数据路径 | `--test_path /data/teco-data/ISIC2018/test/` |
| `--result_path` | 结果保存路径 | `--result_path ./result/` |
| `--cuda_idx` | 使用的GPU索引 | `--cuda_idx 1` |
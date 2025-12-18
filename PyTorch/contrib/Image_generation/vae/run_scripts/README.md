## 参数介绍

| 参数名              | 说明                    | 示例                  |
| ---------------- | --------------------- | ------------------- |
| `--batch-size`   | 每个 batch 的训练样本数量      | `--batch-size 128`  |
| `--epochs`       | 训练的总 epoch 数          | `--epochs 1`        |
| `--no-accel`     | 禁用加速器（如 AMP 等），默认使用加速 | `--no-accel`        |
| `--seed`         | 随机种子，确保实验可复现          | `--seed 1`          |
| `--log-interval` | 每隔多少个 batch 记录一次训练日志  | `--log-interval 10` |

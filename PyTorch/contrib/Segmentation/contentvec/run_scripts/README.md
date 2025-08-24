## 参数介绍
| 参数名                                  | 说明                  | 示例                                                                   |
| ------------------------------------ | ------------------- | -------------------------------------------------------------------- |
| `--expdir`                           | 实验目录，用于保存日志和检查点     | `--expdir ./exp`                                                     |
| `--config_dir`                       | Hydra 配置文件所在目录      | `--config_dir ./contentvec/config/contentvec`                        |
| `--config_name`                      | Hydra 配置文件名         | `--config_name contentvec`                                           |
| `--task.data`                        | 数据集 metadata 路径     | `--task.data /data/teco-data/LibriSpeech/metadata`                   |
| `--task.label_dir`                   | 标签文件目录              | `--task.label_dir /data/teco-data/LibriSpeech/label`                 |
| `--task.labels`                      | 使用的标签列表             | `--task.labels ["km"]`                                               |
| `--task.spk2info`                    | spk2info 字典路径       | `--task.spk2info /data/teco-data/LibriSpeech/metadata/spk2info.dict` |
| `--task.crop`                        | 是否裁剪输入数据            | `--task.crop true`                                                   |
| `--dataset.train_subset`             | 训练数据子集名称            | `--dataset.train_subset train`                                       |
| `--dataset.num_workers`              | DataLoader 工作进程数    | `--dataset.num_workers 10`                                           |
| `--dataset.max_tokens`               | 每个 batch 最大 token 数 | `--dataset.max_tokens 500000`                                        |
| `--optimization.update_freq`         | 梯度累积更新频率            | `--optimization.update_freq [1]`                                     |
| `--optimization.max_update`          | 最大训练步数              | `--optimization.max_update 200`                                      |
| `--lr_scheduler.warmup_updates`      | 学习率预热步数             | `--lr_scheduler.warmup_updates 8000`                                 |
| `--common.log_interval`              | 日志打印间隔（步数）          | `--common.log_interval 1`                                            |
| `--model.label_rate`                 | 标签采样率               | `--model.label_rate 50`                                              |
| `--model.encoder_layers_1`           | Encoder 层数          | `--model.encoder_layers_1 3`                                         |
| `--model.logit_temp_ctr`             | 对比学习温度参数            | `--model.logit_temp_ctr 0.1`                                         |
| `--model.ctr_layers`                 | 用于对比学习的层索引          | `--model.ctr_layers [-6]`                                            |
| `--model.extractor_mode`             | 特征提取模式              | `--model.extractor_mode default`                                     |
| `--checkpoint.keep_best_checkpoints` | 保留的最佳 checkpoint 数  | `--checkpoint.keep_best_checkpoints 10`                              |
| `--criterion.loss_weights`           | 损失函数权重              | `--criterion.loss_weights [10,1e-5]`                                 |


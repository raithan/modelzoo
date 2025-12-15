## 参数介绍
| 参数名                             | 说明                       | 示例                                       |
| ------------------------------- | ------------------------ | ---------------------------------------- |
| `--task_name`                   | 要训练的任务名称                 | `--task_name ner`                        |
| `--data_dir`                    | 输入数据目录，应包含训练数据文件         | `--data_dir ./data/`                     |
| `--model_type`                  | 模型类型                     | `--model_type bert`                      |
| `--model_name_or_path`          | 预训练模型路径或模型名              | `--model_name_or_path bert-base-uncased` |
| `--output_dir`                  | 模型预测结果与检查点保存目录           | `--output_dir ./outputs`                 |
| `--markup`                      | 标签标注方式                   | `--markup bios`                          |
| `--loss_type`                   | 损失函数类型                   | `--loss_type ce`                         |
| `--config_name`                 | 配置文件路径（可选）               | `--config_name ./config.json`            |
| `--tokenizer_name`              | 分词器路径（可选）                | `--tokenizer_name bert-base-uncased`     |
| `--cache_dir`                   | 缓存预训练模型的目录               | `--cache_dir ./cache`                    |
| `--train_max_seq_length`        | 训练时最大输入序列长度              | `--train_max_seq_length 128`             |
| `--eval_max_seq_length`         | 验证时最大输入序列长度              | `--eval_max_seq_length 512`              |
| `--do_train`                    | 是否进行训练                   | `--do_train`                             |
| `--do_eval`                     | 是否进行验证                   | `--do_eval`                              |
| `--do_predict`                  | 是否进行测试预测                 | `--do_predict`                           |
| `--evaluate_during_training`    | 是否在训练过程中进行评估             | `--evaluate_during_training`             |
| `--do_lower_case`               | 是否使用小写模型（如 uncased BERT） | `--do_lower_case`                        |
| `--do_adv`                      | 是否启用对抗训练                 | `--do_adv`                               |
| `--adv_epsilon`                 | 对抗扰动的 epsilon 值          | `--adv_epsilon 1.0`                      |
| `--adv_name`                    | 对抗训练中指定的嵌入层名             | `--adv_name word_embeddings`             |
| `--per_gpu_train_batch_size`    | 每个GPU上的训练 batch size     | `--per_gpu_train_batch_size 8`           |
| `--per_gpu_eval_batch_size`     | 每个GPU上的验证 batch size     | `--per_gpu_eval_batch_size 8`            |
| `--gradient_accumulation_steps` | 梯度累积步数                   | `--gradient_accumulation_steps 1`        |
| `--learning_rate`               | Adam 优化器的初始学习率           | `--learning_rate 5e-5`                   |
| `--crf_learning_rate`           | CRF 或 Linear 层的学习率       | `--crf_learning_rate 5e-5`               |
| `--weight_decay`                | 权重衰减                     | `--weight_decay 0.01`                    |
| `--adam_epsilon`                | Adam 优化器的 epsilon        | `--adam_epsilon 1e-8`                    |
| `--max_grad_norm`               | 最大梯度裁剪阈值                 | `--max_grad_norm 1.0`                    |
| `--num_train_epochs`            | 总训练 epoch 数              | `--num_train_epochs 3.0`                 |
| `--max_steps`                   | 总训练步数（覆盖 epoch 设置）       | `--max_steps 10000`                      |
| `--warmup_proportion`           | 线性预热比例                   | `--warmup_proportion 0.1`                |
| `--logging_steps`               | 日志记录间隔（步）                | `--logging_steps 50`                     |
| `--save_steps`                  | 模型保存间隔（步）                | `--save_steps 50`                        |
| `--eval_all_checkpoints`        | 是否评估所有 checkpoint        | `--eval_all_checkpoints`                 |
| `--predict_checkpoints`         | 指定某 checkpoint 用于预测      | `--predict_checkpoints 1000`             |
| `--no_cuda`                     | 不使用 CUDA，即使可用            | `--no_cuda`                              |
| `--overwrite_output_dir`        | 覆盖已有的输出目录内容              | `--overwrite_output_dir`                 |
| `--overwrite_cache`             | 覆盖缓存的数据集                 | `--overwrite_cache`                      |
| `--seed`                        | 初始化的随机种子                 | `--seed 42`                              |
| `--fp16`                        | 启用混合精度训练（16-bit）         | `--fp16`                                 |
| `--fp16_opt_level`              | AMP 优化等级（O0\~O3）         | `--fp16_opt_level O1`                    |
| `--local_rank`                  | 分布式训练用的本地rank            | `--local_rank 0`                         |
| `--server_ip`                   | 用于远程调试的 IP               | `--server_ip 127.0.0.1`                  |
| `--server_port`                 | 用于远程调试的端口号               | `--server_port 8888`                     |

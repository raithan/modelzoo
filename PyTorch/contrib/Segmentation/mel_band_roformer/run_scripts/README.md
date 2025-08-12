## **模型与训练参数**

| 参数名                          | 说明                                                                                                     | 示例                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| `--model_type`               | 指定模型类型，可选：`mdx23c`、`htdemucs`、`segm_models`、`mel_band_roformer`、`bs_roformer`、`swin_upernet`、`bandit`。 | `--model_type mel_band_roformer`               |
| `--config_path`              | 模型配置文件路径（YAML/JSON）。                                                                                   | `--config_path configs/mel_band_roformer.yaml` |
| `--start_check_point`        | 训练初始 checkpoint（权重文件）路径，可选。                                                                            | `--start_check_point weights/last.ckpt`  |
| `--results_path`             | 存放训练结果（权重、日志、指标）的目录。                                                                                   | `--results_path results/`                |
| `--device_ids`               | 使用的 GPU ID 列表（多 GPU 用空格分隔）。                                                                            | `--device_ids 0 1`                       |
| `--seed`                     | 随机数种子，保证可复现性。                                                                                          | `--seed 42`                              |
| `--train_lora`               | 启用 LoRA 训练模式。                                                                                          | `--train_lora`                           |
| `--lora_checkpoint`          | LoRA 权重初始 checkpoint 路径。                                                                               | `--lora_checkpoint weights/lora_init.pt` |
| `--use_standard_loss`        | 对 Roformer 模型，使用外部指定的 loss 而非内部定义的。                                                                    | `--use_standard_loss`                    |
| `--save_weights_every_epoch` | 每个 epoch 保存一次权重文件。                                                                                     | `--save_weights_every_epoch`             |

---

## **数据与加载参数**

| 参数名              | 说明                                                                                                                  | 示例                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `--data_path`    | 训练数据路径，可提供多个文件夹。                                                                                                    | `--data_path /data/musdb/train /data/extra` |
| `--dataset_type` | 数据集类型（1\~4），详见 [官方文档](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md)。 | `--dataset_type 1`                          |
| `--valid_path`   | 验证集数据路径，可提供多个文件夹。                                                                                                   | `--valid_path /data/musdb/valid`            |
| `--num_workers`  | DataLoader 线程数。                                                                                                     | `--num_workers 8`                           |
| `--pin_memory`   | DataLoader 是否固定内存（提升 GPU 数据传输效率）。                                                                                   | `--pin_memory`                              |

---

## **损失函数参数**

| 参数名                       | 说明                                                                                                                          | 示例                            |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `--loss`                  | 使用的损失函数列表，可组合：`masked_loss`、`mse_loss`、`l1_loss`、`multistft_loss`、`spec_masked_loss`、`spec_rmse_loss_coef`、`log_wmse_loss`。 | `--loss masked_loss mse_loss` |
| `--masked_loss_coef`      | `masked_loss` 权重系数。                                                                                                         | `--masked_loss_coef 1.0`      |
| `--mse_loss_coef`         | `mse_loss` 权重系数。                                                                                                            | `--mse_loss_coef 0.5`         |
| `--l1_loss_coef`          | `l1_loss` 权重系数。                                                                                                             | `--l1_loss_coef 1.0`          |
| `--log_wmse_loss_coef`    | `log_wmse_loss` 权重系数。                                                                                                       | `--log_wmse_loss_coef 0.1`    |
| `--multistft_loss_coef`   | `multistft_loss` 权重系数。                                                                                                      | `--multistft_loss_coef 0.001` |
| `--spec_masked_loss_coef` | `spec_masked_loss` 权重系数。                                                                                                    | `--spec_masked_loss_coef 1.0` |
| `--spec_rmse_loss_coef`   | `spec_rmse_loss_coef` 权重系数。                                                                                                 | `--spec_rmse_loss_coef 1.0`   |

---

## **验证与指标参数**

| 参数名                      | 说明                                                                                                   | 示例                           |
| ------------------------ | ---------------------------------------------------------------------------------------------------- | ---------------------------- |
| `--pre_valid`            | 在训练前先运行一次验证。                                                                                         | `--pre_valid`                |
| `--metrics`              | 训练/验证时计算的指标，可选：`sdr`、`l1_freq`、`si_sdr`、`log_wmse`、`aura_stft`、`aura_mrstft`、`bleedless`、`fullness`。 | `--metrics sdr si_sdr`       |
| `--metric_for_scheduler` | 用于学习率调度的指标，必须在 `--metrics` 中包含。                                                                      | `--metric_for_scheduler sdr` |
| `--each_metrics_in_name` | 保存 checkpoint 文件名时仅包含人声的指标。                                                                          | `--each_metrics_in_name`     |

---

## **WandB 与日志参数**

| 参数名           | 说明                                        | 示例                  |
| ------------- | ----------------------------------------- | ------------------- |
| `--wandb_key` | Weights & Biases API Key（启用 WandB 记录时使用）。 | `--wandb_key xxxxx` |
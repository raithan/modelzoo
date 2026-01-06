## **模型与训练参数**

| 参数名                           | 说明                                          | 示例                               |
| ----------------------------- | ------------------------------------------- | -------------------------------- |
| `--conf-file`                 | 模型配置文件路径，指定网络结构、优化器、学习率等信息。                 | `--conf-file configs/yolov6n.py` |
| `--epochs`                    | 训练的总轮数（epoch 数）。                            | `--epochs 400`                   |
| `--batch-size`                | 所有 GPU 的总 batch size。若多 GPU，会自动按 GPU 数量拆分。  | `--batch-size 32`                |
| `--bs_per_gpu`                | 每张 GPU 上的 batch size，用于自动调整学习率（常用于大模型如 P6）。 | `--bs_per_gpu 16`                |
| `--device`                    | 指定训练设备，例如 `0`（单卡）、`0,1,2,3`（多卡）或 `cpu`。     | `--device 0,1,2,3`               |
| `--gpu_count`                 | 指定使用 GPU 的数量（在部分多卡训练或 DDP 模式下使用）。           | `--gpu_count 4`                  |
| `--local_rank`                | DDP 参数，用于多进程训练时自动传入（无需手动修改）。                | `--local_rank 0`                 |
| `--dist_url`                  | 分布式训练的初始化 URL，一般保持 `env://` 默认值。            | `--dist_url env://`              |
| `--resume`                    | 是否从上一次训练的 checkpoint 恢复训练。                  | `--resume`                       |
| `--stop_aug_last_n_epoch`     | 在最后 n 个 epoch 停止强数据增强（如 Mosaic、MixUp）。      | `--stop_aug_last_n_epoch 15`     |
| `--save_ckpt_on_last_n_epoch` | 在最后 n 个 epoch 都保存权重（即使不是最优）。                | `--save_ckpt_on_last_n_epoch 10` |
| `--fuse_ab`                   | 在训练过程中融合 AB 分支（用于双分支网络）。                    | `--fuse_ab`                      |

---

## **数据与加载参数**

| 参数名              | 说明                                            | 示例                           |
| ---------------- | --------------------------------------------- | ---------------------------- |
| `--data-path`    | 数据集配置文件路径（通常为 `.yaml` 文件，定义 train、val 路径及类别）。 | `--data-path data/coco.yaml` |
| `--img-size`     | 训练与验证图像的输入尺寸（像素）。                             | `--img-size 640`             |
| `--workers`      | DataLoader 的线程数量（影响数据加载速度）。                   | `--workers 8`                |
| `--check-images` | 初始化数据集时检查图片是否损坏。                              | `--check-images`             |
| `--check-labels` | 初始化数据集时检查标签文件是否匹配且合法。                         | `--check-labels`             |

---

## **验证与评估参数**

| 参数名                  | 说明                                           | 示例                      |
| -------------------- | -------------------------------------------- | ----------------------- |
| `--eval-interval`    | 每多少个 epoch 进行一次验证。                           | `--eval-interval 20`    |
| `--eval-final-only`  | 只在最后一个 epoch 进行验证。                           | `--eval-final-only`     |
| `--heavy-eval-range` | 在最后 n 个 epoch 每次都进行验证（配合 `--eval-interval`）。 | `--heavy-eval-range 50` |

---

## **蒸馏与量化参数**

| 参数名                    | 说明                              | 示例                                               |
| ---------------------- | ------------------------------- | ------------------------------------------------ |
| `--distill`            | 是否启用知识蒸馏（student 从 teacher 学习）。 | `--distill`                                      |
| `--distill_feat`       | 是否蒸馏特征图（feature map）。           | `--distill_feat`                                 |
| `--teacher_model_path` | teacher 模型路径。                   | `--teacher_model_path weights/yolov6_teacher.pt` |
| `--temperature`        | 蒸馏温度参数，用于 soft label 平滑。        | `--temperature 20`                               |
| `--quant`              | 是否启用量化训练（QAT）。                  | `--quant`                                        |
| `--calib`              | 是否进行量化前的校准阶段（PTQ 校准模式）。         | `--calib`                                        |

---

## **输出与日志参数**

| 参数名                     | 说明                                       | 示例                          |
| ----------------------- | ---------------------------------------- | --------------------------- |
| `--output-dir`          | 保存训练结果（模型、日志、图片）的路径。                     | `--output-dir ./runs/train` |
| `--name`                | 实验名称，保存在 `output_dir/name` 下。            | `--name yolov6_exp1`        |
| `--write_trainbatch_tb` | 每个 epoch 向 TensorBoard 写入一批训练图像（略微降低速度）。 | `--write_trainbatch_tb`     |
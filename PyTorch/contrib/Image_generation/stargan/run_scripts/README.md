## 参数介绍

| 参数名                  | 说明                           | 示例                       |
| -------------------- | ---------------------------- | ------------------------ |
| `--c_dim`            | 第一个数据集（如 CelebA）的标签维度（即属性数）。 | `--c_dim 5`              |
| `--c2_dim`           | 第二个数据集（如 RaFD）的标签维度（即表情数）。   | `--c2_dim 8`             |
| `--celeba_crop_size` | CelebA 图像裁剪尺寸。               | `--celeba_crop_size 178` |
| `--rafd_crop_size`   | RaFD 图像裁剪尺寸。                 | `--rafd_crop_size 256`   |
| `--image_size`       | 最终图像输入网络的尺寸（宽高相等）。           | `--image_size 128`       |
| `--g_conv_dim`       | 生成器中第一层卷积的通道数（后续层会加倍）。       | `--g_conv_dim 64`        |
| `--d_conv_dim`       | 判别器中第一层卷积的通道数（后续层会加倍）。       | `--d_conv_dim 64`        |
| `--g_repeat_num`     | 生成器中的残差块数量。                  | `--g_repeat_num 6`       |
| `--d_repeat_num`     | 判别器中使用的步幅卷积层数量（决定输出特征图缩小比例）。 | `--d_repeat_num 6`       |
| `--lambda_cls`       | 分类损失的权重（主要针对判别器的多类分类任务）。     | `--lambda_cls 1`         |
| `--lambda_rec`       | 重建损失的权重（保证生成图像保留原有信息）。       | `--lambda_rec 10`        |
| `--lambda_gp`        | 判别器中梯度惩罚项的权重（用于 WGAN-GP）。    | `--lambda_gp 10`         |

---

## 训练相关参数

| 参数名                 | 说明                                      | 示例                                                             |
| ------------------- | --------------------------------------- | -------------------------------------------------------------- |
| `--dataset`         | 选择使用的数据集，可选项有 `CelebA`、`RaFD` 或 `Both`。 | `--dataset CelebA`                                             |
| `--batch_size`      | 批大小，即每次迭代所用图像的数量。                       | `--batch_size 16`                                              |
| `--num_iters`       | 判别器训练总迭代次数。                             | `--num_iters 200000`                                           |
| `--num_iters_decay` | 学习率开始衰减的迭代次数（从这个点开始线性衰减至 0）。            | `--num_iters_decay 100000`                                     |
| `--g_lr`            | 生成器的初始学习率。                              | `--g_lr 0.0001`                                                |
| `--d_lr`            | 判别器的初始学习率。                              | `--d_lr 0.0001`                                                |
| `--n_critic`        | 每次更新生成器前，判别器更新的次数。                      | `--n_critic 5`                                                 |
| `--beta1`           | Adam 优化器中 `beta1` 参数。                   | `--beta1 0.5`                                                  |
| `--beta2`           | Adam 优化器中 `beta2` 参数。                   | `--beta2 0.999`                                                |
| `--resume_iters`    | 从指定迭代次数恢复训练（可选）。                        | `--resume_iters 100000`                                        |
| `--selected_attrs`  | 训练 CelebA 时选择的属性（列表形式）。                 | `--selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young` |

---

## 测试相关参数

| 参数名            | 说明                                    | 示例                    |
| -------------- | ------------------------------------- | --------------------- |
| `--test_iters` | 在测试阶段加载的模型对应的训练迭代数（即加载哪个 checkpoint）。 | `--test_iters 200000` |

---

## 杂项参数

| 参数名                 | 说明                              | 示例                       |
| ------------------- | ------------------------------- | ------------------------ |
| `--num_workers`     | DataLoader 加载数据的线程数。            | `--num_workers 4`        |
| `--mode`            | 运行模式：训练或测试。可选 `train` 或 `test`。 | `--mode train`           |
| `--use_tensorboard` | 是否启用 TensorBoard 记录训练指标（布尔值）。   | `--use_tensorboard True` |

---

## 文件与目录参数

| 参数名                  | 说明                      | 示例                                              |
| -------------------- | ----------------------- | ----------------------------------------------- |
| `--celeba_image_dir` | CelebA 数据集图像的路径。        | `--celeba_image_dir /data/teco-data/celeba-stargan/celeba/images`        |
| `--attr_path`        | CelebA 属性标签文件路径。        | `--attr_path /data/teco-data/celeba-stargan/celeba/list_attr_celeba.txt` |
| `--rafd_image_dir`   | RaFD 图像数据路径。            | `--rafd_image_dir data/RaFD/train`              |
| `--log_dir`          | TensorBoard 日志保存路径。     | `--log_dir stargan/logs`                        |
| `--model_save_dir`   | 模型 checkpoint 保存路径。     | `--model_save_dir stargan/models`               |
| `--sample_dir`       | 保存中间生成图像（sample）的路径。    | `--sample_dir stargan/samples`                  |
| `--result_dir`       | 最终生成图像（test 阶段结果）的保存路径。 | `--result_dir stargan/results`                  |

---

## 日志与保存频率参数

| 参数名                 | 说明                       | 示例                        |
| ------------------- | ------------------------ | ------------------------- |
| `--log_step`        | 每训练多少步打印一次日志。            | `--log_step 10`           |
| `--sample_step`     | 每训练多少步保存一次样本图像。          | `--sample_step 1000`      |
| `--model_save_step` | 每训练多少步保存一次模型 checkpoint。 | `--model_save_step 10000` |
| `--lr_update_step`  | 每训练多少步更新一次学习率（如果处于衰减阶段）。 | `--lr_update_step 1000`   |
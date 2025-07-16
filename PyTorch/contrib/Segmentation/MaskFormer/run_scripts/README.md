## 参数介绍

| 参数名                             | 说明                                                                         | 示例                                                             |
| ------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `config`                        | 配置文件路径，必须参数，指定训练使用的配置文件。                                                   | `./configs/maskformer/maskformer_r50-d32_8xb2-160k_ade20k-512x512.py`                            |
| `--work-dir`                    | 保存日志和模型的目录路径。若不指定，则默认使用配置文件中的路径。                                           | `--work-dir work_dirs/my_experiment`                           |
| `--resume`                      | 是否自动从 `work_dir` 中最近的 checkpoint 恢复训练。                                     | `--resume`                                                     |
| `--amp`                         | 是否启用自动混合精度训练（Automatic Mixed Precision, AMP），用于加速训练并减少显存占用。                | `--amp`                                                        |
| `--cfg-options`                 | 用于覆盖配置文件中的某些配置项，支持键值对格式 `key=value`，列表可用逗号或括号表示。                           | `--cfg-options model.backbone.depth=50 data.samples_per_gpu=8` |
| `--launcher`                    | 指定分布式训练的启动器类型。可选值：`none`（默认，不分布式）、`pytorch`、`slurm`、`mpi`。                 | `--launcher pytorch`                                           |
| `--local_rank` / `--local-rank` | 当前进程的本地 rank（由分布式训练框架自动传入），在多卡训练时指定。兼容 `--local-rank` 和 `--local_rank` 参数。 | `--local_rank 0`                                               |

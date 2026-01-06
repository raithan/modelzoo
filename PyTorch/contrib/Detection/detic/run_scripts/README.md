## 参数介绍

| 参数名                                  | 说明                                                       | 默认值           | 示例                                                                                 |
| :----------------------------------- | :------------------------------------------------------- | :------------ | :--------------------------------------------------------------------------------- |
| **config**                           | 训练配置文件路径。通常为 `.py` 或 `.yaml` 格式的配置文件，定义模型结构、数据集、优化器等内容。  | 必填            | `configs/eva/eva-l-p14_8xb16_in1k-196px.py`                                        |
| **--work-dir**                       | 指定保存日志和模型权重的目录。如果不指定，将使用配置文件中定义的路径。                      | `None`        | `--work-dir ./work_dirs/eva_l`                                                     |
| **--amp**                            | 启用自动混合精度（Automatic Mixed Precision, AMP）训练，以加速训练并降低显存占用。 | `False`       | `--amp`                                                                            |
| **--auto-scale-lr**                  | 是否根据批大小自动调整学习率（Linear Scaling Rule）。                     | `False`       | `--auto-scale-lr`                                                                  |
| **--resume [path/auto]**             | 恢复训练。如果不加参数则自动加载最近的检查点 (`auto`)，也可手动指定文件路径。              | `None`        | `--resume auto` 或 `--resume work_dirs/latest.pth`                                  |
| **--cfg-options**                    | 动态覆盖配置文件中部分字段。以 `key=value` 格式传入，支持嵌套结构。                 | `None`        | `--cfg-options train_dataloader.dataset.data_root=../data new_key="[(1,2),(3,4)]"` |
| **--launcher**                       | 启动方式，用于分布式训练。可选：`none`、`pytorch`、`slurm`、`mpi`。          | `none`        | `--launcher pytorch`                                                               |
| **--local_rank**                     | 当前 GPU 的本地 rank，由分布式启动器自动传入（一般无需手动设置）。                   | `0`           | `--local_rank 0`                                                                   |
| **--nnodes**                         | 分布式训练的节点（机器）总数。                                          | `1`           | `--nnodes 1`                                                                       |
| **--nproc-per-node**                 | 每个节点（机器）上的 GPU 数量，即每台机器启动的进程数。                           | 取决于硬件         | `--nproc-per-node 8`                                                               |
| **--node-rank**                      | 当前节点在集群中的编号（从 0 开始）。                                     | `0`           | `--node-rank 0`                                                                    |
| **--master-addr**                    | 主节点（rank 0 节点）的 IP 地址，用于节点间通信。                           | `"127.0.0.1"` | `--master-addr 192.168.1.10`                                                       |
| **--master-port**                    | 主节点通信端口。                                                 | `29500`       | `--master-port 29501`                                                              |
| **--no-validate** *(可选扩展)*           | 若指定，则在训练过程中跳过验证步骤。                                       | `False`       | `--no-validate`                                                                    |
| **--no-pin-memory** *(可选扩展)*         | 若指定，则禁用 DataLoader 的 `pin_memory` 以减少内存锁定。               | `False`       | `--no-pin-memory`                                                                  |
| **--no-persistent-workers** *(可选扩展)* | 若指定，则不启用 `persistent_workers`，在小型数据集时可避免多进程缓存问题。         | `False`       | `--no-persistent-workers`                                                          |

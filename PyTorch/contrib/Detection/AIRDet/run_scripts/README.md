| 参数                  | 作用/含义                                                      | 用法示例                                         |
| ------------------- | ---------------------------------------------------------- | -------------------------------------------- |
| `-f, --config_file` | 指定**配置文件路径**。通常为模型结构、训练超参数、数据集路径等的yaml文件。                  | `-f configs/airdet.yaml`                     |
| `--local_rank`      | 分布式训练中的**本地进程编号**，由分布式启动器自动分配。每个进程一个GPU，代码中用来选GPU设备。       | `--local_rank 0`（DDP时自动传入）                   |
| `--resume`          | 是否**恢复断点训练**。加上此参数后，会自动从最近的checkpoint继续训练。                 | `--resume`                                   |
| `-c, --ckpt`        | 指定**checkpoint文件路径**，即恢复训练或finetune时加载的权重文件。               | `-c weights/latest.pth`                      |
| `opts`              | 用于**命令行动态覆盖配置项**（如超参数、路径等）。是一个接收剩余参数的列表，格式通常是`KEY VALUE`对。 | `img_size 640 batch_size 16 epochs 50`（末尾追加） |
# 运行run_airdets.py示例：python -m torch.distributed.launch --nproc_per_node=1 run_airdets.py -f configs/airdet_s.py


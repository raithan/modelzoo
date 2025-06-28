## 参数介绍

参数名 | 解释 | 样例
-----------------|-----------------|-----------------
config | 配置文件。 | --config  ../configs/resnet/resnet18_8xb16_cifar10.py
work-dir | 工作目录。 | --work-dir ./work_dirs/resnet18_8xb16_cifar10
resume | 恢复训练。 | --resume auto
amp | 是否使用amp。 | --amp
no-validate | 是否验证。 | --no-validate
auto-scale-lr | 是否自动调整学习率。 | --auto-scale-lr
no-pin-memory | 是否使用pin_memory。 | --no-pin-memory
no-persistent-workers | 是否使用persistent_workers。 | --no-persistent-workers
cfg-options | 动态覆盖配置文件。 | --cfg-options train_dataloader.dataset.data_root=xxx
launcher | 启动方式。 | --launcher pytorch
local_rank | 本地rank。 | --local_rank 0
nnodes | 节点数。 | --nnodes 1
nproc-per-node | 每个节点的进程数。 | --nproc-per-node 8
master-addr | master地址。 | --master-addr 127.0.0.1
master-port | master端口。 | --master-port 29500
node-rank | 节点rank。 | --node-rank 0
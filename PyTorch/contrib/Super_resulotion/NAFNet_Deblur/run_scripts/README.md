## 参数介绍

参数名 | 说明 | 示例
-----------------|-----------------|-----------------
opt | 配置 YAML 文件的路径。 | --opt options/train/GoPro/NAFNet-width32.yml
launcher |指定使用 PyTorch 原生的分布式启动器 | --launcher pytorch
nproc_per_node | DDP时，每个node上的rank数量。不输入时，默认为1，表示单SPA运行。 | --nproc_per_node 1
node_rank|多机时，node的序号（由torchrun自动注入）。|--node_rank 0
master_port|多机时，主节点的端口号。|--master_port 4321
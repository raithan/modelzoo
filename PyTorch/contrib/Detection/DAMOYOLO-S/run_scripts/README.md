参数名 | 说明 | 示例
-----------------|-----------------|-----------------
-m |启动分布式 | -m torch.distributed.launch
--nproc_per_node |指定每个节点上启动的进程数 | --nproc_per_node=4
-f| 配置文件路径| -f configs/damoyolo_tinynasL25_S.py


示例： 单机四卡训练
python -m torch.distributed.launch --nproc_per_node=4 run_DAMOYOLO_S.py -f configs/damoyolo_tinynasL25_S.py 
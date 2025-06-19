## 参数介绍

参数名 | 说明 | 示例
-----------------|-----------------|-----------------
-m |启动分布式 | -m torch.distributed.launch
--nproc_per_node |指定每个节点上启动的进程数 | --nproc_per_node=4
--use_env| 分布式启动脚本自动从环境变量读取一些关键的分布式训练配置参数| --use_env
--coco_path| 数据集路径 | --coco_path /data/teco-data/COCO 
--output_dir |输出结果路径| --output_dir output/conddetr_r50_epoch50

示例： python -m torch.distributed.launch --nproc_per_node=4 --use_env run_ConditionalDETR.py --coco_path /data/teco-data/COCO --output_dir output/conddetr_r50_epoch50
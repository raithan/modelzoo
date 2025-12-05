## 参数介绍

参数名 | 解释 | 样例
-----------------|-----------------|-----------------
task	| 任务类型	| captioning (图像描述生成)
lr_sched  |	学习率调度器  |	linear_warmup_cosine_lr
init_lr	| 初始学习率	| 1e-5
min_lr	|最小学习率	 | 0
warmup_lr	| 预热学习率	| 1e-8
warmup_steps	| 预热步数	| 1000
weight_decay  |	权重衰减  |	0.05
max_epoch  |	最大训练轮数  |	5
batch_size_train  |	训练批次大小  |	1
accum_grad_iters  | 	梯度累积步数  |	16 (有效批次大小: 16)
batch_size_eval  |	评估批次大小  |	4
num_workers  |	数据加载工作进程数  |	0
max_len |	生成文本最大长度  |	30
min_len |	生成文本最小长度 |	8
num_beams |	Beam Search宽度 |	5
seed |	随机种子 |	42
output_dir |	输出目录 |	"output/BLIP2/Caption_coco"
amp |	自动混合精度 |	True
resume_ckpt_path |	恢复训练检查点路径 |	null
evaluate |	仅评估模式 |	False
device |	训练设备 |	"sdaa"
world_size |	分布式训练世界大小 |	1
distributed |	是否分布式训练 |	True
## 数据集划分参数
参数名	解释	配置值
train_splits	训练集划分	["train"]
valid_splits	验证集划分	["val"]
test_splits	测试集划分	["test"]
日志参数
参数名	| 解释	|  配置值
-----------------|-----------------|-----------------
log_freq  |	常规日志频率  |	0
extra_log_freq |	额外统计窗口频率  |	0
## 分布式训练参数
参数名 |	解释 |	配置值
-----------------|-----------------|-----------------
dist_url |	分布式训练URL |	"env://"
world_size |	进程总数 |	1



## 参数介绍

参数名 | 解释 | 样例
-----------------|-----------------|-----------------
script_path	当前脚本所在路径，自动获取。	script_path=$(dirname $(readlink -f "$0"))
bpe_dir	BPE（Byte Pair Encoding）文件目录。	--bpe-dir=../../utils/BPE
user_dir	自定义模块目录。	--user-dir=../../ofa_module
data_dir	数据集存放路径。	--data_dir=/data/teco-data/ofa/dataset/refcoco_data
data	训练和验证数据文件路径。	--data=${data_dir}/refcoco_train.tsv,${data_dir}/refcoco_val.tsv
selected_cols	从数据文件中选择的列索引。	--selected-cols=0,4,2,3
restore_file	预训练模型权重文件路径。	--restore-file=/data/teco-data/ofa/ofa_large.pt
task	任务类型。	--task=refcoco
arch	模型架构名称。	--arch=ofa_large
criterion	损失函数类型。	--criterion=adjust_label_smoothed_cross_entropy
label_smoothing	标签平滑系数。	--label-smoothing=0.1
lr	学习率。	--lr=3e-5
max_epoch	最大训练轮数。	--max-epoch=1
batch_size	批次大小。	--batch-size=2
update_freq	梯度累积步数。	--update-freq=8
warmup_ratio	学习率预热比例。	--warmup-ratio=0.06
resnet_drop_path_rate	ResNet的drop path率。	--resnet-drop-path-rate=0.0
encoder_drop_path_rate	编码器drop path率。	--encoder-drop-path-rate=0.2
decoder_drop_path_rate	解码器drop path率。	--decoder-drop-path-rate=0.2
dropout	dropout率。	--dropout=0.1
attention_dropout	注意力dropout率。	--attention-dropout=0.0
max_src_length	源序列最大长度。	--max-src-length=80
max_tgt_length	目标序列最大长度。	--max-tgt-length=20
num_bins	位置编码的bin数量。	--num-bins=1000
patch_image_size	图像patch大小。	--patch-image-size=128
prompt_type_method	提示类型方法。	--encoder-prompt-type=prefix
encoder_prompt_length	编码器提示长度。	--encoder-prompt-length=20
decoder_prompt_length	解码器提示长度。	--decoder-prompt-length=20
encoder_prompt	是否启用编码器提示。	--encoder-prompt
decoder_prompt	是否启用解码器提示。	--decoder-prompt
weight_decay	权重衰减系数。	--weight-decay=0.01
optimizer	优化器类型。	--optimizer=adam
adam_betas	Adam优化器的beta参数。	--adam-betas="(0.9,0.999)"
adam_eps	Adam优化器的epsilon参数。	--adam-eps=1e-08
clip_norm	梯度裁剪的范数阈值。	--clip-norm=1.0
lr_scheduler	学习率调度策略。	--lr-scheduler=polynomial_decay
fixed_validation_seed	验证集的固定随机种子。	--fixed-validation-seed=7
keep_best_checkpoints	保留的最佳检查点数量。	--keep-best-checkpoints=1
save_interval	保存间隔（轮数）。	--save-interval=1
validate_interval	验证间隔（轮数）。	--validate-interval=1
save_interval_updates	保存间隔（更新步数）。	--save-interval-updates=500
validate_interval_updates	验证间隔（更新步数）。	--validate-interval-updates=500
eval_acc	是否启用准确率评估。	--eval-acc
eval_args	评估参数配置。	--eval-args='{"beam":5,"min_len":4,"max_len_a":0,"max_len_b":4}'
best_checkpoint_metric	最佳检查点评估指标。	--best-checkpoint-metric=score
maximize_best_checkpoint_metric	是否最大化最佳指标。	--maximize-best-checkpoint-metric
fp16	是否启用混合精度训练。	--fp16
fp16_scale_window	FP16缩放窗口大小。	--fp16-scale-window=512
find_unused_parameters	是否查找未使用的参数。	--find-unused-parameters
add_type_embedding	是否添加类型嵌入。	--add-type-embedding
scale_attn	是否缩放注意力。	--scale-attn
scale_fc	是否缩放全连接层。	--scale-fc
scale_heads	是否缩放头。	--scale-heads
disable_entangle	是否禁用纠缠。	--disable-entangle
num_workers	数据加载的线程数。	--num-workers=0
MASTER_PORT	分布式训练通信端口。	export MASTER_PORT=6051
CUDA_VISIBLE_DEVICES	使用的GPU设备ID。	CUDA_VISIBLE_DEVICES=0
log_dir	训练日志保存目录。	log_dir=./refcoco_logs
save_dir	模型检查点保存目录。	save_dir=./refcoco_checkpoints

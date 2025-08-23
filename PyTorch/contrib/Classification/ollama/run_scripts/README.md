## 参数介绍

参数名 | 解释 | 默认值
--------------------------------------------------------------------------
model_path  |预训练模型路径。                  |	/data/teco-data/ollama/Chinese-Mistral-7B-Instruct-v0.1	
dataset_path | 训练数据集路径（JSON格式）。     |	/data/teco-data/ruozhiba/ruozhiba_qa.json	
output_dir | 输出目录，用于保存训练结果和检查点。|	/output/Mistral-7B-ruozhiba	
log_file | 训练日志文件路径。                   |	sdaa.log	
name | 数据集名称标识。                         |	ruozhiba	
per_device_train_batch_size | 每个设备的训练批次大小。|	2	
gradient_accumulation_steps | 梯度累积步数。    |	2	
num_train_epochs            | 训练轮数。        |	2	
learning_rate | 学习率。                        |	1e-4	
save_steps | 保存检查点的步数间隔。              |	25
save_total_limit | 最多保存的检查点数量。	     |  2
max_length | 序列最大长度。                  	| 384	
lora_r | LoRA的秩。	                            | 8	
lora_alpha | LoRA的alpha参数。	                | 32	
lora_dropout | LoRA的dropout率。	            | 0.1	
fp16 | 是否使用FP16混合精度训练。	             | True	
gradient_checkpointing | 是否使用梯度检查点节省内存。	|  True	
logging_steps | 日志记录步数间隔。	             | 1	
device | 训练设备（自动检测）。	                 | auto	

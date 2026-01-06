from datasets import Dataset
import pandas as pd
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MistralForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import torch, os, json, time, torch_sdaa
from peft import LoraConfig, TaskType, get_peft_model
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning)  # 忽略告警

# 定义TCAP格式日志回调类
class TCAPLoggerCallback(TrainerCallback):
    """
    TCAP格式日志输出回调类，记录train.loss, train.ips, train.total_time
    格式: TCAPPDLL YYYY-MM-DD HH:MM:SS.ffffff - Epoch: X Iteration: Y  rank : Z  train.loss : A  train.ips : B imgs/s train.total_time : C
    """
    def __init__(self, interval=1, log_file="tcap.log"):
        self.interval = interval
        self.log_file = log_file
        self.last_time = time.time()
        self.start_time = time.time()
        self.step_start_time = time.time()
        
        # 创建日志文件目录
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        
        # 创建日志文件并写入头部
        with open(log_file, 'w') as f:
            f.write("# TCAP Style Training Log\n")
            f.write("# Format: TCAPPDLL YYYY-MM-DD HH:MM:SS.ffffff - Epoch: X Iteration: Y rank : Z train.loss : A train.ips : B imgs/s train.total_time : C\n\n")
    
    def on_train_begin(self, args, state, control, **kwargs):
        print("\n" + "="*50)
        print("TCAP Training Started")
        print("Log file: {}".format(self.log_file))
        print("="*50 + "\n")
        self.start_time = time.time()
        self.last_time = time.time()
        
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.interval != 0:
            return
        
        now = time.time()
        step_time = now - self.step_start_time
        
        # 如果能获取loss则记录，否则跳过
        try:
            loss = state.log_history[-1].get('loss', 0.0)
        except (IndexError, AttributeError):
            loss = 0.0
        
        # 估算每秒处理的样本数 (IPS: Images/Samples Per Second)
        batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        ips = batch_size / step_time if step_time > 0 else 0
        
        # 获取当前设备rank
        rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # 获取当前时间，精确到微秒
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        
        # TCAP格式日志
        tcap_log = f"TCAPPDLL {current_time} - Epoch: {int(state.epoch)} Iteration: {state.global_step}"
        tcap_log += f"  rank : {rank}  train.loss : {loss}  train.ips : {ips} imgs/s train.total_time : {step_time} "
        
        # 打印日志
        print(tcap_log)
        
        # 写入日志文件
        with open(self.log_file, 'a') as f:
            f.write(tcap_log + "\n")
        
        self.step_start_time = now
    
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        print("\n" + "="*50)
        print(f"Training completed. Total time: {total_time:.2f}s")
        print(f"Final step: {state.global_step}, Final loss: {state.log_history[-1].get('loss', 'N/A')}")
        print("="*50 + "\n")


# 检查transformers版本并做相应的修补
transformers_version = transformers.__version__
print(f"Transformers版本: {transformers_version}")

# 修补Trainer类以兼容旧版本
if hasattr(Trainer, "_wrap_model"):
    original_wrap_model = Trainer._wrap_model
    
    def patched_wrap_model(self, model, training=True, dataloader=None):
        """修补_wrap_model方法以删除keep_torch_compile参数"""
        try:
            # 尝试使用原始方法
            return original_wrap_model(self, model, training, dataloader)
        except TypeError as e:
            if "keep_torch_compile" in str(e):
                # 如果错误涉及keep_torch_compile参数，使用自定义实现
                print("应用兼容性补丁来解决keep_torch_compile错误...")
                # 基于transformers旧版本的实现
                if self.accelerator.unwrap_model(model) is not model:
                    return self.accelerator.unwrap_model(model)
                return model
            else:
                raise  # 重新抛出其他类型的错误
    
    # 应用补丁
    Trainer._wrap_model = patched_wrap_model
    print("已应用Trainer._wrap_model补丁以兼容旧版本transformers")

device = 'sdaa' if torch.sdaa.is_available() else 'cpu'
# 模型文件路径
model_path = r'/data/teco-data/ollama/Chinese-Mistral-7B-Instruct-v0.1'
# 训练过程数据保存路径
name = 'ruozhiba'  # 只使用名称部分
dataset_path = '/data/teco-data/ruozhiba/ruozhiba_qa.json'  # 完整的数据集路径
output_dir = f'./output/Mistral-7B-{name}'
log_file = f'sdaa.log'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

print(f"Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model path: {model_path}")
print(f"Dataset path: {dataset_path}")
print(f"Output directory: {output_dir}")
print(f"Log file: {log_file}")

# 检查并修复配置文件
config_path = os.path.join(model_path, 'config.json')
if os.path.exists(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 如果模型类型不是mistral，修改它
        if 'model_type' not in config or config['model_type'].lower() != 'mistral':
            config['model_type'] = 'mistral'
            
            # 备份原始文件
            os.system(f'cp {config_path} {config_path}.backup')
            
            # 写入修改后的配置
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"已修复配置文件: {config_path}")
    except Exception as e:
        print(f"检查配置文件时出错: {e}")


# 加载tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


#加载数据集
print("Loading dataset...")
df = pd.read_json(dataset_path)  # 使用完整路径
ds = Dataset.from_pandas(df)
print(ds)

# 对数据集进行处理，需要将数据集的内容按大模型的对话格式进行处理
def process_func_mistral(example):
    MAX_LENGTH = 384  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    instruction = tokenizer(
        f"<s>[INST] <<SYS>>\n\n<</SYS>>\n\n{example['instruction']+example['input']}[/INST]",add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为pad_token_id咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

print("Processing dataset...")
inputs_id = ds.map(process_func_mistral, remove_columns=ds.column_names)
print("Attempting to load model with AutoModelForCausalLM...")
# 删除旧模型对象并清理显存
try:
    del model
except NameError:
    pass

if torch.sdaa.is_available():
    torch.sdaa.empty_cache()
    device = 'sdaa'
else:
    torch.cuda.empty_cache()  # 如果是 CUDA 环境
    device = 'cpu'

# 尝试加载 AutoModelForCausalLM
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.bfloat16, use_cache=False
    )
except (KeyError, ValueError):
    # 如果失败则尝试 MistralForCausalLM
    model = MistralForCausalLM.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.bfloat16, use_cache=False
    )


print("Model loaded successfully:")
print(model)

print("Enabling gradient checkpointing...")
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

print("Configuring LoRA...")
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)

print("Applying LoRA adapter to model...")
model = get_peft_model(model, config)
model.print_trainable_parameters()

print("Setting up training arguments...")
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    logging_steps=1,  # 每步都记录日志
    num_train_epochs=2,
    save_steps=25,
    save_total_limit=2,
    fp16=True,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    # 添加这一行来记录loss
    # report_to=["tensorboard"],
    report_to=[],
)

print("Creating trainer...")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=inputs_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 添加TCAP格式日志回调
tcap_logger = TCAPLoggerCallback(interval=1, log_file=log_file)
trainer.add_callback(tcap_logger)
trainer.train()

print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model saved to: {output_dir}")
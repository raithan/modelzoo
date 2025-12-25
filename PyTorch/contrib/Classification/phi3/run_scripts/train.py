from datasets import Dataset
from torch.sdaa import amp
import pandas as pd
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
import torch
import os
import time
from datetime import datetime
from peft import LoraConfig, TaskType, get_peft_model
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 修改 Trainer 类以解决兼容性问题
class CustomTrainer(Trainer):
    def _wrap_model(self, model, training=True, dataloader=None):
        if self.accelerator.unwrap_model(model) is not model:
            return self.accelerator.unwrap_model(model)
        return model

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
            f.write(f"# Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Transformers version: {transformers.__version__}\n\n")
    
    def on_train_begin(self, args, state, control, **kwargs):
        print("\n" + "="*50)
        print("TCAP Training Started")
        print(f"Log file: {self.log_file}")
        print(f"Transformers version: {transformers.__version__}")
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
        
        try:
            loss = state.log_history[-1].get('loss', 0.0)
        except (IndexError, AttributeError):
            loss = 0.0
        
        batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        ips = batch_size / step_time if step_time > 0 else 0
        
        rank = int(os.environ.get("LOCAL_RANK", 0))
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        
        tcap_log = f"TCAPPDLL {current_time} - Epoch: {int(state.epoch)} Iteration: {state.global_step}"
        tcap_log += f"  rank : {rank}  train.loss : {loss:.4f}  train.ips : {ips:.2f} imgs/s train.total_time : {step_time:.4f}"
        
        print(tcap_log)
        
        with open(self.log_file, 'a') as f:
            f.write(tcap_log + "\n")
        
        self.step_start_time = now
    
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        final_log = f"\n{'='*50}\n"
        final_log += f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        final_log += f"Total training time: {total_time:.2f}s\n"
        final_log += f"Final step: {state.global_step}\n"
        final_log += f"Final loss: {state.log_history[-1].get('loss', 'N/A')}\n"
        final_log += f"{'='*50}\n"
        
        print(final_log)
        with open(self.log_file, 'a') as f:
            f.write(final_log)

def main():
    print(f"Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Transformers version: {transformers.__version__}")
    
    # 创建输出目录
    os.makedirs("./output/Phi-3", exist_ok=True)
    
    # Data loading and preprocessing
    print("\nLoading dataset...")
    df = pd.read_json('/data/teco-data/phi3/datasets/huanhuan.json')
    ds = Dataset.from_pandas(df)
    print(f"Dataset loaded: {len(ds)} examples")
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        '/data/teco-data/phi3/Phi-3-mini-4k-instruct', 
        use_fast=False, 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer initialized")
    
    def process_func(example):
        MAX_LENGTH = 384
        instruction = tokenizer(
            f"<|user|>\n{example['instruction'] + example['input']}<|end|>\n<|assistant|>\n",
            add_special_tokens=False
        )
        response = tokenizer(f"{example['output']}<|end|>\n", add_special_tokens=False)
        
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
        
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    print("\nProcessing dataset...")
    tokenized_id = ds.map(
        process_func,
        remove_columns=ds.column_names,
        desc="Processing dataset"
    )
    print("Dataset processing completed")
    
    # Model initialization
    print("\nInitializing model...")
    model = AutoModelForCausalLM.from_pretrained(
        '/data/teco-data/phi3/Phi-3-mini-4k-instruct',
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        trust_remote_code=True
    )
    model.train()
    model.enable_input_require_grads()
    print("Model initialized")
    
    # LoRA configuration
    print("\nConfiguring LoRA...")
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    # Training configuration
    print("\nSetting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./output/Phi-3",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=1,
        num_train_epochs=1,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        fp16=True,
        ddp_find_unused_parameters=False,
        report_to=[],
        remove_unused_columns=False,
    )
    
    # Initialize trainer with CustomTrainer
    print("\nInitializing trainer...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt"
        ),
    )
    
    # Add TCAP logger
    tcap_logger = TCAPLoggerCallback(
        interval=1,
        log_file="sdaa.log"
    )
    trainer.add_callback(tcap_logger)
    
    # Start training
    print("\nStarting training...")
    trainer.train()
    
    # Save model and tokenizer
    print("\nSaving model and tokenizer...")
    lora_path = './Phi-3_lora'
    trainer.model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"Model and tokenizer saved to: {lora_path}")
    
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise
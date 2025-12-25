from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, LoraConfig, TaskType

def load_model(model_path, lora_path):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # LoRA configuration
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)
    
    return model, tokenizer

def generate_response(prompt, model, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        eos_token_id=tokenizer.encode('<|endoftext|>')[0]
    )
    
    outputs = generated_ids.tolist()[0][len(model_inputs[0]):]
    response = tokenizer.decode(outputs).split('<|end|>')[0]
    
    return response

if __name__ == "__main__":
    model_path = '/root/autodl-tmp/LLM-Research/Phi-3-mini-4k-instruct'
    lora_path = './Phi-3_lora'
    
    model, tokenizer = load_model(model_path, lora_path)
    
    # Example usage
    prompt = "你是谁？"
    response = generate_response(prompt, model, tokenizer)
    print(response)
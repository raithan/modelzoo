# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelArguments:
    """
    与模型/配置相关的参数
    """
    model_path: str = field(
        default=None,
        metadata={"help": "预训练模型的路径或huggingface.co/models中的模型标识符"}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "是否使用基于tokenizers库的快速分词器"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "加载模型和分词器时是否信任远程代码"}
    )
    device: str = field(
        default='sdaa',
        metadata={"help": "用于训练的设备，例如'sdaa'、'cuda'、'cpu'"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "要应用LoRA的模块名称列表"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA秩，较低的秩产生更小的模型，但可能影响质量"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha参数"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA层的dropout概率"}
    )


@dataclass
class DataArguments:
    """
    与输入数据相关的参数
    """
    dataset_path: str = field(
        default=None,
        metadata={"help": "训练数据集JSON文件的路径"}
    )
    max_seq_length: int = field(
        default=384,
        metadata={"help": "分词的最大序列长度。超过此长度的序列将被截断"}
    )


@dataclass
class TrainingArguments:
    """
    与训练方式相关的参数
    """
    output_dir: str = field(
        default="./output",
        metadata={"help": "模型预测和检查点将写入的输出目录"}
    )
    log_file: str = field(
        default="./output/sdaa_training.log",
        metadata={"help": "TCAP格式日志文件的路径"}
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "每个GPU/TPU/SDAA核心/CPU用于训练的批量大小"}
    )
    gradient_accumulation_steps: int = field(
        default=2,
        metadata={"help": "在执行反向/更新传递之前要累积的更新步骤数"}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "优化器的初始学习率"}
    )
    num_train_epochs: int = field(
        default=2,
        metadata={"help": "要执行的训练轮数总数"}
    )
    save_steps: int = field(
        default=25,
        metadata={"help": "每X个更新步骤保存一次检查点"}
    )
    save_total_limit: int = field(
        default=2,
        metadata={"help": "限制检查点的总数。删除较旧的检查点"}
    )
    logging_steps: int = field(
        default=1,
        metadata={"help": "每X个更新步骤记录一次日志"}
    )
    report_to: List[str] = field(
        default_factory=lambda: ["tensorboard"],
        metadata={"help": "报告结果和日志的集成列表"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "使用梯度检查点以牺牲较慢的反向传递为代价节省内存"}
    )
    save_on_each_node: bool = field(
        default=True,
        metadata={"help": "在进行多节点分布式训练时，在每个节点上保存模型检查点"}
    )
    train_with_checkpoint: bool = field(
        default=True,
        metadata={"help": "如果可用，是否从最新的检查点恢复训练"}
    )
    log_interval: int = field(
        default=1,
        metadata={"help": "TCAP日志记录的间隔（以步骤为单位）"}
    )


def get_args():
    """
    获取微调的默认参数。
    您可以根据需要修改这些默认值或传递命令行参数。
    """
    model_args = ModelArguments(
        model_path=r'/data/teco-data/ollama/Chinese-Mistral-7B-Instruct-v0.1'
    )
    
    data_args = DataArguments(
        dataset_path='/data/teco-data/ruozhiba/ruozhiba_qa.json'
    )
    
    name = 'ruozhiba'
    
    training_args = TrainingArguments(
        output_dir=f'./output/Mistral-7B-{name}',
        log_file=f'./output/sdaa_{name}.log'
    )
    
    return model_args, data_args, training_args


def parse_args():
    """解析命令行参数并与默认值结合"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mistral模型的微调脚本")
    
    # 添加模型参数
    parser.add_argument("--model_path", type=str, help="预训练模型的路径")
    parser.add_argument("--device", type=str, help="使用的设备（sdaa, cuda, cpu）")
    
    # 添加数据参数
    parser.add_argument("--dataset_path", type=str, help="数据集的路径")
    parser.add_argument("--max_seq_length", type=int, help="最大序列长度")
    
    # 添加训练参数
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, help="每设备批量大小")
    
    # 解析参数
    args = parser.parse_args()
    
    # 获取默认参数
    model_args, data_args, training_args = get_args()
    
    # 如果提供了命令行参数，则覆盖默认值
    for k, v in vars(args).items():
        if v is not None:
            if hasattr(model_args, k):
                setattr(model_args, k, v)
            elif hasattr(data_args, k):
                setattr(data_args, k, v)
            elif hasattr(training_args, k):
                setattr(training_args, k, v)
    
    return model_args, data_args, training_args
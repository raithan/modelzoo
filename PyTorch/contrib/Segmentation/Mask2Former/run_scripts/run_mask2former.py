#BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
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

from argument import parse_options
import os
import subprocess
from pathlib import Path

def build_hyper_parameters(args):
    # 提取并存储参数
    amp = args.amp
    cfg_options = args.cfg_options

    hyper_parameters = ""

    if args.amp:
        hyper_parameters += f" --amp"
    if args.cfg_options:
        cfg_str = ""
        for k, v in args.cfg_options.items():
            # 确保值被正确格式化
            if isinstance(v, str) and ' ' in v:
                # 如果值包含空格，用引号包裹
                cfg_str += f"{k}='{v}' "
            else:
                cfg_str += f"{k}={v} "       
        hyper_parameters += f" --cfg-options {cfg_str.strip()} "

    return hyper_parameters

def build_train_command(args, hyper_parameters):
    """构建训练命令字符串"""
    # 构建基本命令
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--master_port={args.master_port}",
        "../tools/train.py",  # 你可以修改为其他脚本路径
        args.config,       # 配置文件路径
    ]

    # 添加 launcher
    cmd.append("--launcher")
    cmd.append(args.launcher)

    return " ".join(cmd) + " " + hyper_parameters


if __name__ == '__main__':
    # 解析参数
    args = parse_options()
    
    # 构建并执行命令
    hyper_parameters = build_hyper_parameters(args)
    cmd = build_train_command(args, hyper_parameters)
    print(f"Executing command: {cmd}")
    
    try:
        # 使用check_call确保命令执行成功
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        print(f"Command failed with exit code: {exit_code}")
        exit(exit_code)

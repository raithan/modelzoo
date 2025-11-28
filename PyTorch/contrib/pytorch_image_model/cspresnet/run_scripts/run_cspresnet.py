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

from argument import _parse_args
import os
import subprocess
from pathlib import Path

def build_train_command(args):
    """构建训练命令字符串"""
    os.chdir("..")

    cmd = [
        "torchrun",
        "--nproc_per_node=4",
        "train.py",
        "--data-dir", str(args.data_dir),
        "--model", str(args.model),
        "--sched", str(args.sched),
        "--epochs", str(args.epochs),
        "--warmup-epochs", str(args.warmup_epochs),
        "--lr", str(args.lr),
        "--reprob", str(args.reprob),
        "--remode", str(args.remode),
        "--batch-size", str(args.batch_size),
    ]

    # bool 参数：只有为 True 时才加入
    if args.amp:
        cmd.append("--amp")

    # workers
    cmd.extend(["-j", str(args.workers)])

    cmd.extend(["--log-interval", str(args.log_interval)])

    # 返回命令字符串
    return " ".join(cmd) + " "

if __name__ == '__main__':
    # 解析参数
    args, args_text = _parse_args()
    
    # 构建并执行命令
    cmd = build_train_command(args)
    print(f"Executing command: {cmd}")
    
    try:
        # 使用check_call确保命令执行成功
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        print(f"Command failed with exit code: {exit_code}")
        exit(exit_code)

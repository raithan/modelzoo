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

def build_train_command(args):
    """构建训练命令字符串"""
    # 创建实验目录
    os.chdir("..")
    print(os.getcwd())
    Path(args.expdir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-u", "./fairseq/fairseq_cli/hydra_train.py",
        "--config-dir", args.config_dir,
        "--config-name", args.config_name,
    ] + [f"{k}={v}" for k, v in vars(args).items()
         if k not in ["config_dir", "config_name", "expdir"]] + [
        f"hydra.run.dir={args.expdir}"]

    return cmd

if __name__ == '__main__':
    # 解析参数
    args = parse_options()
    
    # 构建命令
    cmd = build_train_command(args)
    print(f"Executing command: HYDRA_FULL_ERROR=1 {' '.join(cmd)}")
    
    try:
        subprocess.check_call("HYDRA_FULL_ERROR=1 " + " ".join(cmd), shell=True)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        print(f"Command failed with exit code: {exit_code}")
        exit(exit_code)


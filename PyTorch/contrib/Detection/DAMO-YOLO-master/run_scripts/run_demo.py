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

import os
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run distributed training demo")
    parser.add_argument('--nproc_per_node', type=int, default=1, help='Number of processes per node (number of GPUs)') # 分布式训练
    parser.add_argument('--train_file', type=str, default='tools/train.py', help='Path to training script')
    parser.add_argument('-f', '--config', type=str, default='configs/damoyolo_tinynasL25_S.py', help='Config file')
    parser.add_argument('--extra_args', type=str, default='', help='Extra args for train script')
    return parser.parse_args()

def main():
    args = parse_args()

    # 构建命令
    cmd = (
        f"python -m torch.distributed.launch "
        f"--nproc_per_node={args.nproc_per_node} "
        f"{args.train_file} -f {args.config} {args.extra_args}"
    )

    print(f"[INFO] Executing command:\n{cmd}")

    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code: {e.returncode}")
        exit(e.returncode)

if __name__ == '__main__':
    main()

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

import argparse

def parse_options():
    parser = argparse.ArgumentParser('NAFNet training', add_help=False)
    
    # 必需参数
    parser.add_argument('--opt', type=str, required=True, 
                      help='Path to config YAML file')
    
    # 分布式参数组
    dist_group = parser.add_argument_group('Distributed')
    dist_group.add_argument('--launcher', 
                          choices=['none', 'pytorch', 'slurm'], 
                          default='pytorch')  # 默认改为pytorch
    dist_group.add_argument('--master_port', 
                          type=int, 
                          default=4321,
                          help='Master port for distributed training')
    dist_group.add_argument('--nproc_per_node', 
                          type=int, 
                          default=1,
                          help='GPUs per node')
    
    # 隐藏参数（由torchrun自动注入）
    parser.add_argument('--local_rank', 
                      type=int, 
                      default=0,
                      help=argparse.SUPPRESS)  # 对用户隐藏
    
    return parser.parse_args()

if __name__ == "__main__":
    sys.exit(0)
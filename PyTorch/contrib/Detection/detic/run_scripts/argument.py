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
import argparse
from mmengine.config import Config, ConfigDict, DictAction
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector (custom launcher wrapper)')
    
    # ---- 基础参数 ----
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')

    # ---- 可选控制参数 ----
    parser.add_argument('--amp', action='store_true', default=False,
                        help='enable automatic-mixed-precision training')
    parser.add_argument('--auto-scale-lr', action='store_true',
                        help='enable automatically scaling LR.')
    parser.add_argument('--resume', nargs='?', type=str, const='auto',
                        help='resume from checkpoint (path or auto)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help=('override some settings in the used config, '
                              'key-value pairs like key=value or key="[a,b]"'))

    # ---- 分布式参数 ----
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='pytorch', help='job launcher type')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0,
                        help='local rank, passed automatically by torch.distributed')
    
    # ✅ 新增：支持命令中直接传 nnodes, nproc_per_node 等参数（方便 build_command）
    parser.add_argument('--nnodes', type=int, default=1, help='number of nodes for distributed training')
    parser.add_argument('--nproc_per_node', type=int, default=1, help='number of processes per node')
    parser.add_argument('--node_rank', type=int, default=0, help='rank of the current node')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='master node address')
    parser.add_argument('--master_port', type=int, default=29500, help='master node port')

    args = parser.parse_args()

    # ---- 环境变量修正（MMEngine 启动时需要）----
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
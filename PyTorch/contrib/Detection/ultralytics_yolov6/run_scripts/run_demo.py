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

from ultralytics import YOLO
from torch_sdaa.utils import cuda_migrate  # 使用torch_sdaa自动迁移方法
from argument import parse_args

if __name__ == '__main__':
    args = parse_args()

    model = YOLO(args.model)

    train_kwargs = {
        'data': args.data,
        'cache': args.cache,
        'imgsz': args.imgsz,
        'epochs': args.epochs,
        'batch': args.batch,
        'close_mosaic': args.close_mosaic,
        'workers': args.workers,
        'device': args.device,
        'optimizer': args.optimizer,
        'patience': args.patience,
        'resume': args.resume,
        'amp': args.amp,
        'fraction': args.fraction,
        'project': args.project,
        'name': args.name,

    }

    # ========== 启动训练
    print(f"\n[INFO] 启动训练：")
    print(f"模型: {args.model}")
    print(f"数据: {args.data}")
    print(f"训练参数: {train_kwargs}")

    model.train(**train_kwargs)

    print('\n[INFO] 训练结束！')

    # ===== 如需添加验证、推理、导出等，可在此处继续扩展
    # val_results = model.val()
    # print(val_results)

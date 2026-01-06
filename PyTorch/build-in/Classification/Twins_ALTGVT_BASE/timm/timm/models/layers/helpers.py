""" Layer/Module Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
from itertools import repeat
# 兼容不同 PyTorch 版本的导入
import torch
if torch.__version__ < '1.8':
    from torch._six import container_abcs
else:
    from collections.abc import Container, Iterable, Mapping, Sequence

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple






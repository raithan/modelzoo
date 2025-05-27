##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models"""
import loguru
from torch import nn

from resnet import ResNet, Bottleneck


def resnest50(**kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs,
    )
    return model


def resnest101(**kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs,
    )
    return model


def resnest200(**kwargs):
    model = ResNet(
        Bottleneck,
        [3, 24, 36, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs,
    )
    return model


def resnest269(**kwargs):
    model = ResNet(
        Bottleneck,
        [3, 30, 48, 8],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs,
    )
    return model


def resnest50_fast(**kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=True,
        **kwargs,
    )
    return model


def resnest101_fast(**kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=True,
        **kwargs,
    )
    return model


class ResNeSt:
    @classmethod
    def create_pretrained(cls, model_name, **kwargs) -> nn.Module:
        model = None
        if model_name == "resnest50":
            model = resnest50(**kwargs)
        elif model_name == "resnest101":
            model = resnest101(**kwargs)
        elif model_name == "resnest200":
            model = resnest200(**kwargs)
        elif model_name == "resnest269":
            model = resnest269(**kwargs)
        elif model_name == "resnest50_fast":
            model = resnest50_fast(**kwargs)
        elif model_name == "resnest101_fast":
            model = resnest101_fast(**kwargs)
        else:
            raise NotImplementedError
        loguru.logger.info(f"使用 {model_name} 模型")
        return model

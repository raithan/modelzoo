#!/bin/bash

# 运行 train.py 并将标准输出和错误输出重定向到 sdaa.log，同时在终端显示
python train.py 2>&1 | tee sdaa.log

# 运行 loss.py
python loss.py
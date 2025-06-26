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
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def compare_loss(benchmark_loss_array, sdaa_loss_array):
    def MeanRelativeError(cuda_loss, sdaa_loss):
        return ((sdaa_loss - cuda_loss) / cuda_loss).mean()
    def MeanAbsoluteError(cuda_loss, sdaa_loss):
        return (sdaa_loss - cuda_loss).mean()
    benchmark_mean_loss = benchmark_loss_array
    sdaa_mean_loss = sdaa_loss_array
    benchmark_compare_loss = benchmark_mean_loss
    sdaa_compare_loss = sdaa_mean_loss
    mean_relative_error = MeanRelativeError(benchmark_compare_loss, sdaa_compare_loss)
    mean_absolute_error = MeanAbsoluteError(benchmark_compare_loss, sdaa_compare_loss)
    print("MeanRelativeError:", mean_relative_error)
    print("MeanAbsoluteError:", mean_absolute_error)
    if mean_relative_error <= mean_absolute_error:
        print("Rule,mean_relative_error", mean_relative_error)
    else:
        print("Rule,mean_absolute_error", mean_absolute_error)
    print_str = f"{mean_relative_error=} <= 0.05 or {mean_absolute_error=} <= 0.0002"
    if mean_relative_error <= 0.05 or mean_absolute_error <= 0.0002:
        print('pass', print_str)
        return True, print_str
    else:
        print('fail', print_str)
        return False, print_str

def parse_string(string):
    """
    提取日志中的Iteration和train.loss
    返回：迭代次数列表（int）、loss列表（float）
    """
    pattern = r"Iteration: (\d+)\s+train\.loss\s*:\s*([\d\.e\-]+)"
    matches = re.findall(pattern, string)
    iteration_list = [int(m[0]) for m in matches]
    loss_list = [float(m[1]) for m in matches]
    return iteration_list, loss_list

def plot_loss(sdaa_iter, sdaa_loss, a100_iter, a100_loss):
    fig, ax = plt.subplots(figsize=(12, 6))
    # 平滑处理
    if len(sdaa_loss) >= 5:
        sdaa_loss_smoothed = savgol_filter(sdaa_loss, 5, 1)
    else:
        sdaa_loss_smoothed = sdaa_loss
    ax.plot(sdaa_iter, sdaa_loss_smoothed, label="sdaa_loss")
    if len(a100_loss) >= 5:
        a100_loss_smoothed = savgol_filter(a100_loss, 5, 1)
    else:
        a100_loss_smoothed = a100_loss
    ax.plot(a100_iter, a100_loss_smoothed, "--", label="cuda_loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Model Training Loss Curves (Smoothed)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("loss.jpg")
    plt.show()

if __name__=="__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description='modelzoo')
    parser.add_argument('--sdaa-log', type=str, default="sdaa.log")
    parser.add_argument('--cuda-log', type=str, default="cuda.log")
    args = parser.parse_args()
    # sdaa log
    with open(args.sdaa_log, 'r') as f:
        s = f.read()
    sdaa_iter, sdaa_loss = parse_string(s)
    # cuda log
    with open(args.cuda_log, 'r') as f:
        s = f.read()
    a100_iter, a100_loss = parse_string(s)
    # 对齐长度
    length = min(len(a100_loss), len(sdaa_loss))
    sdaa_iter = sdaa_iter[:length]
    sdaa_loss = sdaa_loss[:length]
    a100_iter = a100_iter[:length]
    a100_loss = a100_loss[:length]
    compare_loss(np.array(a100_loss), np.array(sdaa_loss))  # loss误差判断
    plot_loss(sdaa_iter, sdaa_loss, a100_iter, a100_loss)   # loss曲线对比

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
import json  # 用于解析 TCAPPDLL 日志格式
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def compare_loss(benchmark_loss_array, sdaa_loss_array):
    def MeanRelativeError(cuda_loss, sdaa_loss):
        return ((sdaa_loss - cuda_loss) / cuda_loss).mean()

    def MeanAbsoluteError(cuda_loss, sdaa_loss):
        return (sdaa_loss - cuda_loss).mean()

    benchmark_compare_loss = benchmark_loss_array
    sdaa_compare_loss = sdaa_loss_array
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

# 修改后的 parse_string：从 TCAPPDLL JSON 日志中提取 'train.loss'
def parse_string(string):
    """
    从 TCAPPDLL JSON 日志中提取 'train.loss' 的值列表
    """
    matches = []
    lines = string.strip().splitlines()
    for line in lines:
        if not line.startswith("TCAPPDLL"):
            continue
        try:
            json_str = line.split("TCAPPDLL")[1].strip()
            log_entry = json.loads(json_str)
            if log_entry.get("type") == "LOG":
                data_str = log_entry.get("data", "")
                data_dict = eval(data_str) if isinstance(data_str, str) else data_str
                if "train.loss" in data_dict:
                    matches.append(float(data_dict["train.loss"]))
        except Exception as e:
            print(f" 跳过格式异常行: {line[:80]}... 原因: {e}")
    print("提取 train.loss 数量：", len(matches))
    return matches

def parse_loss(ret_list):
    step_num = len(ret_list)
    loss_arr = np.zeros(shape=(step_num,))
    for i, loss in enumerate(ret_list):
        loss_arr[i] = float(loss)
    print(loss_arr)
    return loss_arr

def plot_loss(sdaa_loss, a100_loss):
    fig, ax = plt.subplots(figsize=(12, 6))

    smoothed_losses = savgol_filter(sdaa_loss, 5, 1)
    x = list(range(len(sdaa_loss)))
    ax.plot(x, smoothed_losses, label="sdaa_loss")

    smoothed_losses = savgol_filter(a100_loss, 5, 1)
    x = list(range(len(a100_loss)))
    ax.plot(x, smoothed_losses, "--", label="cuda_loss")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Model Training Loss Curves (Smoothed)")
    ax.legend()
    plt.savefig("loss.jpg")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Compare loss between CUDA and SDAA')
    parser.add_argument('--sdaa-log', type=str, default="sdaa.log")
    parser.add_argument('--cuda-log', type=str, default="cuda.log")
    args = parser.parse_args()

    with open(args.sdaa_log, 'r') as f:
        sdaa_log_str = f.read()
    with open(args.cuda_log, 'r') as f:
        cuda_log_str = f.read()

    sdaa_res = parse_string(sdaa_log_str)
    cuda_res = parse_string(cuda_log_str)

    length = min(len(cuda_res), len(sdaa_res))
    sdaa_loss = parse_loss(sdaa_res[:length])
    a100_loss = parse_loss(cuda_res[:length])

    compare_loss(a100_loss, sdaa_loss)
    plot_loss(sdaa_loss, a100_loss)

# BSD 3- Clause License
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def compare_loss(benchmark_loss_array: np.ndarray, sdaa_loss_array: np.ndarray):
    """
    比较两条 loss 曲线，返回 (是否通过, 说明字符串)
    - MeanRelativeError: 仅在基准不为 0 的位置计算
    - MeanAbsoluteError: 简单均值差
    通过条件：mean_relative_error <= 0.05 或 mean_absolute_error <= 0.0002
    """
    # 统一长度
    n = min(len(benchmark_loss_array), len(sdaa_loss_array))
    if n == 0:
        print("fail 无有效数据：两份日志均未解析到 loss 数值")
        return False, "no data"

    a = benchmark_loss_array[:n].astype(float)
    b = sdaa_loss_array[:n].astype(float)

    # 相对误差只在分母非零处计算
    eps = 1e-12
    mask = np.abs(a) > eps
    if mask.any():
        mean_relative_error = np.mean((b[mask] - a[mask]) / a[mask])
    else:
        mean_relative_error = np.nan

    mean_absolute_error = np.mean(b - a)

    print("MeanRelativeError:", mean_relative_error)
    print("MeanAbsoluteError:", mean_absolute_error)

    rule = "mean_relative_error" if (not np.isnan(mean_relative_error) and mean_relative_error <= mean_absolute_error) else "mean_absolute_error"
    print(f"Rule,{rule}", mean_relative_error if rule == "mean_relative_error" else mean_absolute_error)

    print_str = f"{mean_relative_error=} <= 0.05 or {mean_absolute_error=} <= 0.0002"
    if (not np.isnan(mean_relative_error) and mean_relative_error <= 0.05) or (mean_absolute_error <= 0.0002):
        print("pass", print_str)
        return True, print_str
    else:
        print("fail", print_str)
        return False, print_str


def parse_string(text: str):
    """
    从日志中提取 loss 数值（字符串列表）。
    兼容以下样式（空格与大小写不敏感）：
    - TCAPPDLL ... rank : 0  train.loss : 1741.86 ...
    - TCAPPDLL ... rank:0   train.loss_avg: 13.58 ...
    - 不带 TCAPPDLL 前缀的同类行
    """
    # 关键词允许存在或不存在；优先匹配 train.loss / train.loss_avg 的数值
    # 数字模式支持科学计数法
    num = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    patterns = [
        rf"train\.(?:loss|loss_avg)\s*:\s*{num}",
    ]

    matches = []
    for line in text.splitlines():
        for pat in patterns:
            m = re.search(pat, line, flags=re.IGNORECASE)
            if m:
                matches.append(m.group(1))
                break  # 一行命中一次即可
    return matches


def parse_loss(str_list):
    """将字符串列表转为 float ndarray"""
    if not str_list:
        return np.array([], dtype=float)
    try:
        return np.array([float(x) for x in str_list], dtype=float)
    except Exception:
        # 忽略无法转浮点的条目
        out = []
        for x in str_list:
            try:
                out.append(float(x))
            except Exception:
                pass
        return np.array(out, dtype=float)


def _smooth(arr: np.ndarray, window: int = 5, polyorder: int = 1):
    """
    对序列做 Savitzky-Golay 平滑：
    - 自动选择不超过长度且为奇数的窗口
    - 序列太短则直接返回原始数据
    """
    n = len(arr)
    if n < 3:
        return arr
    # 调整窗口长度：不超过 n，且为奇数
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return arr
    if polyorder >= w:
        polyorder = max(1, w - 1)
    try:
        return savgol_filter(arr, w, polyorder)
    except Exception:
        return arr


def plot_loss(sdaa_loss: np.ndarray, a100_loss: np.ndarray, out="loss.jpg"):
    if len(sdaa_loss) == 0 or len(a100_loss) == 0:
        print("跳过绘图：至少一条曲线为空")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    xs = np.arange(len(sdaa_loss))
    ys = _smooth(sdaa_loss.astype(float), 5, 1)
    ax.plot(xs, ys, label="sdaa_loss")

    xa = np.arange(len(a100_loss))
    ya = _smooth(a100_loss.astype(float), 5, 1)
    ax.plot(xa, ya, "--", label="cuda_loss")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Model Training Loss Curves (Smoothed)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out)
    print(f"保存曲线到 {out}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Compare training losses between accelerators")
    parser.add_argument("--sdaa-log", type=str, default="sdaa.log")
    parser.add_argument("--cuda-log", type=str, default="cuda.log")
    parser.add_argument("--out", type=str, default="loss.jpg")
    args = parser.parse_args()

    with open(args.sdaa_log, "r", encoding="utf-8", errors="ignore") as f:
        sdaa_txt = f.read()
    with open(args.cuda_log, "r", encoding="utf-8", errors="ignore") as f:
        cuda_txt = f.read()

    sdaa_res = parse_string(sdaa_txt)
    cuda_res = parse_string(cuda_txt)

    print(f"sdaa 提取到 {len(sdaa_res)} 条 loss")
    print(f"cuda 提取到 {len(cuda_res)} 条 loss")

    n = min(len(sdaa_res), len(cuda_res))
    sdaa_loss = parse_loss(sdaa_res[:n])
    a100_loss = parse_loss(cuda_res[:n])

    compare_loss(a100_loss, sdaa_loss)  # 比较 loss
    plot_loss(sdaa_loss, a100_loss, out=args.out)  # 对比 loss 曲线图
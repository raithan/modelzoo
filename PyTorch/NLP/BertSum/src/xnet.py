import re
import matplotlib.pyplot as plt

def extract_xent_values(log_file_path):
    xent_values = []
    pattern = r'xent:\s*([\d.]+)'  # 正则表达式匹配 xent 后的数字
    
    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                xent = float(match.group(1))
                xent_values.append(xent)
    
    return xent_values

def plot_xent_curves(xent_values1, xent_values2, label1='sdaa', label2='cuda'):
    plt.figure(figsize=(10, 6))
    plt.plot(xent_values1, label=label1)
    plt.plot(xent_values2, label=label2)
    plt.xlabel('Step')
    plt.ylabel('Xent Value')
    plt.title('Xent Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

# 示例用法
log_file_path1 = 'bert_transformer_sdaa'  # 替换为你的第一份日志文件路径
log_file_path2 = 'bert_transformer_cuda'  # 替换为你的第二份日志文件路径

xent_values1 = extract_xent_values(log_file_path1)
xent_values2 = extract_xent_values(log_file_path2)

plot_xent_curves(xent_values1, xent_values2)
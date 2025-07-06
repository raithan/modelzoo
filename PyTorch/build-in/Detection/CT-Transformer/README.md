# FSMN-VAD (中文 16 kHz) ⊕ Torch-SDAA

## 1. 模型概述

**FSMN-VAD (Feed-forward Sequential Memory Network for Voice Activity Detection)**  
由达摩院提出，面向 16 kHz 中文通用场景，可在离线或流式模式下精准分割语音/静音区段。  
本仓库提供一键脚本，将官方 *FunASR-1.2.6* 推理管线迁移到 **Torch-SDAA** 显卡，验证端到端推理流程。

- 论文链接：[[2008.10790\]] FSMN-based Front-End for Online Speech Recognition  
- 模型链接：<https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch>
- FunASR 仓库：<https://github.com/modelscope/FunASR>

---

## 2. 快速开始

完整推理流程如下：

1. **基础环境安装** - 检查 SDAA 驱动 & Python 依赖  
2. **准备音频** - 16 kHz 单声道 WAV  
3. **构建运行环境** - 创建虚拟环境并安装依赖  
4. **执行推理** - 单文件 / 目录批量  
5. **结果查看** - JSON 或终端输出

### 2.1 基础环境安装

| 组件 | 版本示例 | 说明 |
|------|----------|------|
| SDAA Driver/Runtime | 2.1.0 | `sdaa-smi` 检查 |
| PyTorch-SDAA | 2.4.0a0+git4451b0e | 与驱动匹配 |
| Python | ≥3.9 | 推荐 3.10 |
| NumPy | `<2` | 1.26.4，避免 ABI 冲突 |

### 2.2 准备音频

- **采样率**：16 000 Hz  
- **通道数**：1（单声道）  
- 可用 ffmpeg 统一转换：  
  ```bash
  ffmpeg -i input.wav -ar 16000 -ac 1 example.wav

conda create -n funvad python=3.10 -y
conda activate funvad

### 2.3 构建环境

# 安装 torch-sdaa 与匹配 nightly PyTorch
pip install "numpy<2" torch_sdaa==2.1.0 \
            torch==2.4.0a0+git4451b0e -f https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html

# FunASR + ModelScope + 其它依赖
pip install funasr==1.2.6 modelscope==1.11.1 soundfile tqdm

### 2.4 执行推理

git clone https://github.com/your_name/fsmn-vad-sdaa.git
cd fsmn-vad-sdaa
python fsmn_vad_infer.py --wav example.wav #单文件处理
python fsmn_vad_infer.py --wav_dir wav_dir/ --out_dir vad_json/ #批量处理

### 2.5 推理结果

# example.wav 语音段（秒）： [[0.72, 1.98], [2.65, 4.03]]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSMN-VAD (中文 16 kHz 通用) 推理脚本，支持 Torch-SDAA 加速
==============================================================

$ python fsmn_vad_infer.py --wav my.wav
$ python fsmn_vad_infer.py --wav_dir wavs/ --out_dir vad_json/
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

# --- 1. 先导入 torch_sdaa，让 PyTorch 识别新设备 -----------------------------
try:
    import torch_sdaa          # noqa: F401  # 注册自定义后端
    _sdaa_ok = True
except ImportError:
    print("[Warn] torch_sdaa 未安装，将退回 CPU")
    _sdaa_ok = False

import torch
from funasr import AutoModel
import soundfile as sf

# ---------------------------------------------------------------------------


def pick_device() -> str:
    """优先选 sdaa:0；没有就 cpu"""
    if _sdaa_ok and torch.sdaa.device_count():
        return "sdaa:0"
    return "cpu"


def load_model(device: str = "cpu"):
    """加载 FSMN-VAD 推理管线"""
    print(f"[Info] Loading FSMN-VAD to {device} ...")
    model = AutoModel(
        model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        model_revision="v2.0.4",
        device=device,
        disable_update=True,         # 关闭版本检查，加速启动
        log_level="WARNING",
    )
    return model


def infer_one(model, wav: Path):
    speech, sr = sf.read(wav)
    if sr != 16000:
        raise RuntimeError(f"{wav} 采样率为 {sr} Hz，应先转换为 16 kHz 单声道")

    # ★ 关键修改：用 generate 而不是直接调用 model
    return model.generate(input=str(wav))


def save_json(res: Dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="FSMN-VAD inference with Torch-SDAA")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--wav", type=Path,
                   help="单个 16 kHz wav 文件")
    g.add_argument("--wav_dir", type=Path,
                   help="目录批量模式：遍历其中所有 *.wav")
    parser.add_argument("--out_dir", type=Path, default=Path("vad_out"),
                        help="批量模式结果保存目录；默认 vad_out")
    args = parser.parse_args()

    device = pick_device()
    vad_model = load_model(device)

    if args.wav:
        res = infer_one(vad_model, args.wav)
        print(f"# {args.wav.name} 语音段（秒）：", res)
    else:
        wav_paths: List[Path] = sorted(args.wav_dir.glob("*.wav"))
        if not wav_paths:
            raise FileNotFoundError(f"{args.wav_dir} 下没有 WAV 文件")
        for wav in wav_paths:
            res = infer_one(vad_model, wav)
            save_json(res, args.out_dir / f"{wav.stem}.json")
            print(f"[OK] {wav.name} -> {wav.stem}.json")


if __name__ == "__main__":
    main()

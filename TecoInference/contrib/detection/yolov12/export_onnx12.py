#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Export YOLOv12 .pt -> .onnx (no NMS)
- Works even if torch doesn't provide F.scaled_dot_product_attention
- Optional preprocess packing (BGR->RGB & normalize)
- Dynamic batch support
"""

import argparse
import os
import sys
import time
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx


# -----------------------
# 0) PATCH: SDPA fallback
# -----------------------
if not hasattr(F, "scaled_dot_product_attention"):
    # Minimal, numerically-stable-ish fallback; enough to let Ultralytics import.
    def _sdpa(q, k, v, attn_mask=None, dropout_p: float = 0.0, is_causal: bool = False, scale=None):
        # q,k,v: (..., heads, L, d)
        d = q.size(-1)
        scale_val = (d ** 0.5) if (scale is None) else (1.0 / scale)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale_val
        if is_causal:
            # causal mask: allow j <= i
            Lq, Lk = scores.size(-2), scores.size(-1)
            causal = torch.tril(torch.ones(Lq, Lk, device=scores.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal, float("-inf"))
        if attn_mask is not None:
            # assume 1 for keep, 0 for mask-out
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        if dropout_p and probs.requires_grad:
            probs = F.dropout(probs, p=dropout_p)
        return torch.matmul(probs, v)

    F.scaled_dot_product_attention = _sdpa  # type: ignore


# -----------------------
# 1) (Optional) Preprocess wrapper
# -----------------------
class PreprocessWrapper(nn.Module):
    """
    Pack BGR->RGB and normalize into the graph.
    Expect input in BGR uint8/float[0..255] -> convert to RGB float[0..1] then (x-mean)/std.
    """
    def __init__(self, model, rgb_mean=(0.485, 0.456, 0.406), rgb_std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(rgb_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(rgb_std).view(1, 3, 1, 1))

    def forward(self, x):
        # x: N,C,H,W in BGR, uint8 or float[0..255]
        if x.dtype != torch.float32 and x.dtype != torch.float16:
            x = x.float()
        x = x / 255.0
        x = x[:, [2, 1, 0], :, :]  # BGR -> RGB
        x = (x - self.mean) / self.std
        return self.model(x)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, default='../yolov12n.pt', help='Path to YOLOv12 *.pt')
    p.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='H W')
    p.add_argument('--batch-size', type=int, default=1, help='Batch size for dummy input')
    p.add_argument('--half', action='store_true', help='Export with FP16 (model & input). GPU only.')
    p.add_argument('--simplify', action='store_true', help='Run onnx-simplifier after export')
    p.add_argument('--dynamic-batch', action='store_true', help='Dynamic batch axis for input/output')
    p.add_argument('--with-preprocess', action='store_true', help='Pack BGR->RGB & normalize into graph')
    p.add_argument('--opset', type=int, default=13, help='ONNX opset')
    p.add_argument('--device', default='cpu', help='cpu or cuda:0')
    p.add_argument('--fp16_post', action='store_true',
                   help='Convert ONNX to FP16 afterwards (onnxconverter_common). Use when --half is not possible.')
    return p.parse_args()


def main():
    args = parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1
    h, w = args.img_size

    # -----------------------
    # 2) Device & dtype
    # -----------------------
    use_cuda = (args.device != 'cpu') and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else 'cpu')
    if args.half and device.type == 'cpu':
        print("[!] --half 仅在 GPU 可用时生效；将忽略。你可以改用 --fp16_post 在导出后再转 FP16。")
        args.half = False

    # -----------------------
    # 3) Load YOLOv12 from Ultralytics
    #    (keep imports AFTER our SDPA patch)
    # -----------------------
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("✗ 无法导入 ultralytics：", e)
        print("请先: pip install -U ultralytics huggingface_hub safetensors onnx onnxsim")
        sys.exit(1)

    print(f"Loading weights: {args.weights}")
    yolo = YOLO(args.weights)
    model = yolo.model.to(device).eval()

    # -----------------------
    # 4) Optional preprocess pack
    # -----------------------
    if args.with_preprocess:
        model = PreprocessWrapper(model).to(device).eval()

    # -----------------------
    # 5) Dummy input
    # -----------------------
    B = args.batch_size
    x = torch.zeros(B, 3, h, w, device=device)
    if args.half:
        model = model.half()
        x = x.half()

    # Dry run
    with torch.no_grad():
        _ = model(x)

    # -----------------------
    # 6) Dynamic axes
    # -----------------------
    dynamic_axes = None
    input_name = 'images'
    output_names = ['outputs']  # Ultralytics head returns raw predictions tensor list -> traced as a single output

    if args.dynamic_batch:
        dynamic_axes = {
            input_name: {0: 'batch'},
            output_names[0]: {0: 'batch'},
        }
        B = 'batch'  # for printing only

    # -----------------------
    # 7) Export
    # -----------------------
    export_path = os.path.splitext(args.weights)[0] + '.onnx'
    print(f"Exporting to: {export_path}  (opset={args.opset})")
    t0 = time.time()
    onnx_model = None

    try:
        with BytesIO() as f:
            torch.onnx.export(
                model, x, f,
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=True,
                training=torch.onnx.TrainingMode.EVAL,
                input_names=[input_name],
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )
            f.seek(0)
            onnx_model = onnx.load(f)
            onnx.checker.check_model(onnx_model)
    except Exception as e:
        print("✗ ONNX 导出失败：", e)
        sys.exit(2)

    # -----------------------
    # 8) Simplify
    # -----------------------
    if args.simplify:
        try:
            import onnxsim
            print("Running onnxsim.simplify ...")
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "onnxsim check failed"
        except Exception as e:
            print("! onnx-simplifier 失败：", e)

    # -----------------------
    # 9) Optional: convert ONNX to FP16 post-export
    # -----------------------
    if args.fp16_post:
        try:
            from onnxconverter_common import float16
            print("Converting ONNX to FP16 ...")
            onnx_model = float16.convert_float_to_float16(onnx_model)
        except Exception as e:
            print("! FP16 转换失败：", e)

    onnx.save(onnx_model, export_path)
    dt = time.time() - t0
    print(f"✅ Export OK -> {export_path}  (time: {dt:.2f}s)")
    if args.dynamic_batch:
        print(f"   Dynamic batch enabled (0-dim named 'batch').")
    if args.with_preprocess:
        print("   Preprocess packed: BGR->RGB + normalize.")
    if args.half:
        print("   Model exported in FP16 (PyTorch half).")
    elif args.fp16_post:
        print("   Model converted to FP16 (post-export).")


if __name__ == "__main__":
    main()

# BSD 3-Clause License
# (c) Tecorigin Co., Ltd.  All rights reserved.

import os
import sys
import cv2
import argparse
import random
from pathlib import Path
import numpy as np
import torch

# 让脚本能找到 tecoinfer 与 utils
ROOT = Path(__file__).resolve().parents[3]  # .../TecoInference
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from engine.tecoinfer_pytorch import TecoInferEngine
# 与 v6 一致：始终用标准前处理（letterbox + BGR->RGB + /255）和标准后处理
from utils.preprocess.pytorch.yolo_pt import preprocess           # :contentReference[oaicite:3]{index=3}
from utils.postprocess.pytorch.yolo_pt import postprocess

MAX_ENGINE_NUMS = int(os.getenv('MAX_ENGINE_NUMS', 4))

def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ('true','1','yes','y'): return True
    if v in ('false','0','no','n'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

# YOLOv12 -> YOLOv5/6 风格：插 obj=1、必要时做 sigmoid、以及 [B,84,N] 转置
def adapt_yolo12_pred(pred: torch.Tensor) -> torch.Tensor:
    if not isinstance(pred, torch.Tensor):
        pred = torch.as_tensor(pred)

    if pred.ndim == 2:  # [N,C] -> [1,N,C]
        pred = pred.unsqueeze(0)

    if pred.ndim == 3:
        B, A, C = pred.shape
        # [B,84,N] -> [B,N,84]
        if C != 84 and A == 84:
            pred = pred.transpose(1, 2)
            B, A, C = pred.shape

        # [B,N,84] -> 插入 obj=1 -> [B,N,85]
        if C == 84:
            xywh = pred[..., :4]
            cls  = pred[..., 4:]
            obj  = torch.ones((B, A, 1), dtype=pred.dtype, device=pred.device)
            pred = torch.cat([xywh, obj, cls], dim=-1)
            C = 85

        # 若 obj/cls 还不是概率（>1 或 <0），做一次 sigmoid
        probs = pred[..., 4:]
        if (probs.max() > 1.0) or (probs.min() < 0.0):
            pred[..., 4:] = torch.sigmoid(probs)
    return pred

def draw_boxes(img_path: str, results, out_path='output_image.jpg'):
    image = cv2.imread(img_path)
    if image is None:
        print(f"[Warn] cannot read image {img_path}, skip drawing.")
        return
    def rand_color(): return [random.randint(0, 255) for _ in range(3)]
    for det in results:
        for cls_name, boxes in det.items():
            color = rand_color()
            for b in boxes:
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, cls_name, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imwrite(out_path, image)
    print(f"Saved: {out_path}")

def parse_opt():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='./yolov12n.onnx', help='ONNX path')
    p.add_argument('--data-path', type=str, default='./imgs/bus.jpg', help='image path')
    p.add_argument('--input_name', type=str, default='images', help='input tensor name')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--shape', type=int, default=640, help='inference size (pixels)')
    p.add_argument('--conf-thres', type=float, default=0.25)
    p.add_argument('--iou-thres', type=float, default=0.45)
    p.add_argument('--max-det', type=int, default=1000)
    p.add_argument('--target', default='sdaa', help='sdaa or cpu')
    p.add_argument('--half', type=str2bool, default=True, help='use FP16 in engine')
    p.add_argument('--pass_path', type=str, default=None)
    # 与 v6 脚本一致：model_name 只用于引擎构建标签，用哪个都行，但保留 yolov12n 便于区分
    p.add_argument('--model_name', type=str, default='yolov12n',
                   choices=['yolov6n','yolov5','yolov12n'])
    p.add_argument('--save_result', type=str2bool, default=True)
    p.add_argument('--debug', action='store_true')
    opt = p.parse_args()
    opt.dtype = 'float16' if opt.half else 'float32'
    opt.class_name = None  # 与 v6 单图脚本一致（打印时用数字类名）:contentReference[oaicite:4]{index=4}
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    # 构建引擎（与 v6 的单图脚本一致用法）:contentReference[oaicite:5]{index=5}
    input_size = [[max(opt.batch_size // MAX_ENGINE_NUMS, 1), 3, opt.shape, opt.shape]]
    pipeline = TecoInferEngine(
        ckpt=opt.ckpt,
        input_name=opt.input_name,
        target=opt.target,
        model_name=opt.model_name,
        batch_size=opt.batch_size,
        input_size=input_size,
        dtype=opt.dtype,
        pass_path=opt.pass_path
    )

    # **统一走标准前处理**（letterbox + BGR->RGB + /255），与 v6 一致 :contentReference[oaicite:6]{index=6}
    im, padding_shape, image0_shapes = preprocess(
        opt.data_path, opt.batch_size, (opt.shape, opt.shape), half=(opt.dtype=='float16')
    )

    # 推理
    raw = pipeline(im)                   # teco 返回 numpy/torch
    pred = raw[0] if isinstance(raw, (list, tuple)) else raw
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)

    # 适配 YOLOv12 输出到 v5/v6 风格（插 obj=1、sigmoid、转置）
    pred = adapt_yolo12_pred(pred)

    if opt.debug:
        print("pred shape:", tuple(pred.shape))
        print("pred min/max:", float(pred.min()), float(pred.max()))
        print("first row:", pred[0,0,:10])

    # 后处理（与 v6 一致的 yolo_pt.postprocess）
    results = postprocess(
        pred,
        padding_shape,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        max_det=opt.max_det,
        image=im,
        image0_shapes=image0_shapes,
        class_name=opt.class_name
    )

    image_name = os.path.basename(opt.data_path)
    if not results:
        print(f"{image_name}: no detections")
    else:
        print(f"{image_name}:")
        for det in results:
            for k, v in det.items():
                print(k, v)

    if opt.save_result:
        draw_boxes(opt.data_path, results, out_path='output_image.jpg')

    if 'sdaa' in opt.target:
        pipeline.release()

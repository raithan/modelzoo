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

# 搜索路径：.../TecoInference
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from engine.tecoinfer_pytorch import TecoInferEngine
from utils.preprocess.pytorch.yolo_pt import preprocess, IMG_FORMATS
from utils.postprocess.pytorch.yolo_pt import postprocess

MAX_ENGINE_NUMS = int(os.getenv('MAX_ENGINE_NUMS', 4))

def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ('true','1','yes','y'): return True
    if v in ('false','0','no','n'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

# --- YOLOv12 输出自适配：插 obj=1、sigmoid、通道转置 ---
def adapt_yolo12_pred(pred: torch.Tensor) -> torch.Tensor:
    if not isinstance(pred, torch.Tensor):
        pred = torch.as_tensor(pred)
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)               # [N,C] -> [1,N,C]
    if pred.ndim == 3:
        B, A, C = pred.shape
        if C != 84 and A == 84:                # [B,84,N] -> [B,N,84]
            pred = pred.transpose(1, 2)
            B, A, C = pred.shape
        if C == 84:                            # [B,N,84] -> 插入 obj=1 -> [B,N,85]
            xywh = pred[..., :4]
            cls  = pred[..., 4:]
            obj  = torch.ones((B, A, 1), dtype=pred.dtype, device=pred.device)
            pred = torch.cat([xywh, obj, cls], dim=-1)
            C = 85
        probs = pred[..., 4:]                  # 若不是概率，做 sigmoid
        if (probs.max() > 1.0) or (probs.min() < 0.0):
            pred[..., 4:] = torch.sigmoid(probs)
    return pred

def draw_boxes(img_path: str, results, save_dir: Path):
    image = cv2.imread(img_path)
    if image is None:
        print(f"[Warn] cannot read image {img_path}, skip")
        return
    def rand_color(): return [random.randint(0, 255) for _ in range(3)]
    for det in results:
        for cls_name, boxes in det.items():
            color = rand_color()
            for b in boxes:
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, cls_name, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / Path(img_path).name
    cv2.imwrite(str(out), image)

def parse_opt():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='./yolov12n.onnx', help='onnx path')
    p.add_argument('--data-path', type=str, default='./imgs', help='folder of images')
    p.add_argument('--input_name', type=str, default='images', help='input name')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--shape', type=int, default=640)
    p.add_argument('--conf-thres', type=float, default=0.25)
    p.add_argument('--iou-thres', type=float, default=0.45)
    p.add_argument('--max-det', type=int, default=1000)
    p.add_argument('--target', default='sdaa', help='sdaa or cpu')
    p.add_argument('--half', type=str2bool, default=True)
    p.add_argument('--pass_path', type=str, default=None)
    p.add_argument('--model_name', type=str, default='yolov12n', choices=['yolov6n','yolov5','yolov12n'])
    p.add_argument('--save_result', type=str2bool, default=True)
    p.add_argument('--save_dir', type=str, default='./runs_folder')
    p.add_argument('--debug', action='store_true')
    opt = p.parse_args()
    opt.dtype = 'float16' if opt.half else 'float32'
    opt.class_name = None
    return opt

if __name__ == "__main__":
    opt = parse_opt()

    # 构建引擎（同 v6 风格）
    input_size = [[max(opt.batch_size // MAX_ENGINE_NUMS, 1), 3, opt.shape, opt.shape]]
    pipeline = TecoInferEngine(
        ckpt=opt.ckpt, input_name=opt.input_name, target=opt.target,
        model_name=opt.model_name, batch_size=opt.batch_size,
        input_size=input_size, dtype=opt.dtype, pass_path=opt.pass_path
    )

    img_dir = Path(opt.data_path)
    files = sorted([p for p in img_dir.iterdir() if p.suffix.lower().lstrip('.') in IMG_FORMATS])
    if not files:
        print(f"No images in {img_dir}")
    for fp in files:
        im, padding_shape, image0_shapes = preprocess(str(fp), opt.batch_size, (opt.shape, opt.shape), half=(opt.dtype=='float16'))
        raw = pipeline(im)
        pred = raw[0] if isinstance(raw, (list, tuple)) else raw
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        pred = adapt_yolo12_pred(pred)
        if opt.debug:
            print(fp.name, "pred shape:", tuple(pred.shape), "min/max:", float(pred.min()), float(pred.max()))
        results = postprocess(
            pred, padding_shape,
            conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, max_det=opt.max_det,
            image=im, image0_shapes=image0_shapes, class_name=opt.class_name
        )
        if not results:
            print(f"{fp.name}: no detections")
        else:
            print(f"{fp.name}:")
            for det in results:
                for k, v in det.items():
                    print(k, v)
        if opt.save_result:
            draw_boxes(str(fp), results, Path(opt.save_dir))

    if 'sdaa' in opt.target:
        pipeline.release()

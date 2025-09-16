# YOLOv5/6 骨架，适配 tecorigin 硬件（已加 YOLOv12 输出自适配）

import os
import sys
import json
import time
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

sys.path.append(str(Path(__file__).resolve().parents[3]))
from engine.tecoinfer_pytorch import TecoInferEngine

from utils.datasets.yolo_pt_dataset import create_dataloader
from utils.preprocess.pytorch.yolo_pt import preprocess
from utils.postprocess.pytorch.yolo_pt import (
    LOGGER, colorstr, increment_path, scale_boxes, xywh2xyxy, xyxy2xywh, print_args,
    box_iou, ap_per_class, postprocess, TQDM_BAR_FORMAT, CLASSES, CLASSES_ANIMAL
)

MAX_ENGINE_NUMS = int(os.getenv('MAX_ENGINE_NUMS', 4))

def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ('true','1','yes','y'): return True
    if v in ('false','0','no','n'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def check_data(data_path):
    val = os.path.join(data_path, "val2017.txt")
    if not Path(val).exists():
        LOGGER.info(f'\nDataset not found ⚠️, missing {val}')
        raise FileNotFoundError('Dataset not found ❌')
    return val

# --- 和上面一致的 YOLOv12 输出自适配 ---
def adapt_yolo12_pred(pred: torch.Tensor) -> torch.Tensor:
    if not isinstance(pred, torch.Tensor):
        pred = torch.as_tensor(pred)
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    if pred.ndim == 3:
        B, A, C = pred.shape
        if C != 84 and A == 84:
            pred = pred.transpose(1, 2)
            B, A, C = pred.shape
        if C == 84:
            xywh = pred[..., :4]
            cls  = pred[..., 4:]
            obj  = torch.ones((B, A, 1), dtype=pred.dtype, device=pred.device)
            pred = torch.cat([xywh, obj, cls], dim=-1)
        probs = pred[..., 4:]
        if (probs.max() > 1.0) or (probs.min() < 0.0):
            pred[..., 4:] = torch.sigmoid(probs)
    return pred

def process_batch(detections, labels, iouv):
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def save_one_json(predn, jdict, path, class_map):
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])
    box[:, :2] -= box[:, 2:] / 2
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,'category_id': class_map[int(p[5])],'bbox': [round(x,3) for x in b],'score': round(p[4],5)})

def metrics_per_batch(im, preds, targets, paths, shapes, seen, jdict, stats, save_json=False, single_cls=False,):
    class_map = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,
                 39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
    iouv = torch.linspace(0.5, 0.95, 10)
    niou = iouv.numel()
    for si, pred in enumerate(preds):
        labels = targets[targets[:, 0] == si, 1:]
        nl, npr = labels.shape[0], pred.shape[0]
        path, shape = Path(paths[si]), shapes[si][0]
        correct = torch.zeros(npr, niou, dtype=torch.bool)
        seen += 1
        if npr == 0:
            if nl:
                stats.append((correct, *torch.zeros((2, 0)), labels[:, 0]))
            continue
        if single_cls:
            pred[:, 5] = 0
        predn = pred.clone()
        scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])
            scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)
            correct = process_batch(predn, labelsn, iouv)
        stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))
        if save_json:
            save_one_json(predn, jdict, path, class_map)

def compute_metrics(ckpt, dataloader, stats, jdict, seen, data_path, save_dir, data_type, save_json=False, plots=False,):
    class_type = {"coco_yolo_animal": CLASSES_ANIMAL, "coco": CLASSES}
    names = dict(enumerate(class_type[data_type]))
    nc = len(class_type[data_type])

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    ap, ap_class = [], []
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map_ = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        mp = mr = map50 = map_ = 0.0
        tp = fp = ap_class = np.array([])

    nt = np.bincount(stats[3].astype(int), minlength=nc) if len(stats) else np.zeros(nc)
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map_))

    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    if save_json and len(jdict):
        w = ckpt.split("/")[-1]
        anno_json = os.path.join(data_path, 'annotations/instances_val2017.json')
        os.makedirs(save_dir, exist_ok=True)
        pred_json = str(Path(save_dir) / f"{w}_predictions.json")
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            ev = COCOeval(anno, pred, 'bbox')
            ev.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]
            ev.evaluate(); ev.accumulate(); ev.summarize()
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')
    print('eval done.')

def run(model_name="yolov12n", ckpt="./yolov12n.onnx", input_name='images', target='sdaa',
        batch_size=64, shape=640, half=True, data_path='', stride=32, single_cls=False,
        pad=0.5, rect=False, workers=0, conf_thres=0.001, iou_thres=0.6, max_det=300,
        save_json=False, save_dir=Path(''), project=ROOT / 'runs/val', name='exp',
        exist_ok=False, verbose=False, save_engine=False, pass_path=None, data_type="coco", card_bs1=False):

    # 引擎
    input_size = [[max(batch_size // MAX_ENGINE_NUMS, 1), 3, shape, shape]]
    pipeline = TecoInferEngine(
        ckpt=ckpt, input_name=input_name, target=target, model_name=model_name,
        batch_size=batch_size, input_size=input_size, dtype="float16" if half else "float32",
        save_engine=save_engine, pass_path=pass_path, card_bs1=card_bs1
    )

    # dataloader
    val_path = check_data(data_path)
    dataloader = create_dataloader(val_path, shape, batch_size, stride, single_cls, pad=pad, rect=rect, workers=workers,
                                   prefix=colorstr(f'{"val"}: '))[0]

    # 日志目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    from tqdm import tqdm
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)

    seen = 0
    jdict, stats = [], []

    e2e_time, pre_time, run_time, post_time, ips = [], [], [], [], []
    max_step = int(os.environ.get("TECO_INFER_PIPELINES_MAX_STEPS", -1))
    warmup_step = int(os.environ.get("TECO_INFER_PIPELINES_WARMUP_STEPS", 0))
    global_step = 1

    while True:
        for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
            nb, _, h, w = im.shape
            start = time.time()

            # 与 v6 一致：标准前处理
            dealed_im, padding_shape, _ = preprocess(im.numpy(), batch_size, (h, w), half=half)
            t_pre = time.time() - start

            # 推理
            preds = pipeline(dealed_im, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det, batch_padding=True)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            # 适配 YOLOv12 输出
            preds = torch.from_numpy(preds) if isinstance(preds, np.ndarray) else preds
            preds = adapt_yolo12_pred(preds)

            # NMS
            preds = postprocess(preds, padding_shape, conf_thres, iou_thres, labels=[], multi_label=True, max_det=max_det)
            t_e2e = time.time() - start

            # 标注缩放回原图尺寸 & 统计指标
            targets[:, 2:] *= torch.tensor((w, h, w, h))
            metrics_per_batch(im, preds, targets, paths, shapes, seen, jdict, stats, save_json=save_json, single_cls=single_cls)

            if global_step > warmup_step:
                t_post = t_e2e - pipeline.run_time - t_pre
                sps = batch_size / t_e2e
                e2e_time.append(t_e2e); pre_time.append(t_pre); run_time.append(pipeline.run_time); post_time.append(t_post); ips.append(sps)
                if verbose:
                    print(f'e2e: {t_e2e:.4f}s, infer: {pipeline.run_time:.4f}s, preprocess: {t_pre:.4f}s, post: {t_post:.4f}s, sps: {sps:.2f}')

            if max_step > 0 and global_step == max_step:
                break
            global_step += 1
        if global_step >= max_step:
            break

    if 'sdaa' in target:
        pipeline.release()

    # 计算 mAP
    _ = compute_metrics(ckpt, dataloader, stats, jdict, seen, data_path, str(save_dir),
                        data_type, save_json=save_json, plots=False)

    if ips:
        n = len(ips)
        print(f'summary: avg_sps: {sum(ips)/n:.2f} img/s, e2e_time: {sum(e2e_time):.2f}s, '
              f'avg_infer: {sum(run_time[5:])/(max(n-5,1)):.4f}s, avg_pre: {sum(pre_time)/n:.4f}s, avg_post: {sum(post_time)/n:.4f}s')

def parse_opt():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='./yolov12n.onnx')
    p.add_argument('--data-path', type=str, default='/data/teco-data/coco')
    p.add_argument('--data-type', type=str, default='coco')
    p.add_argument('--input_name', type=str, default='images')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--shape', type=int, default=640)
    p.add_argument('--conf-thres', type=float, default=0.001)
    p.add_argument('--iou-thres', type=float, default=0.6)
    p.add_argument('--max-det', type=int, default=300)
    p.add_argument('--target', default='sdaa')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--single-cls', action='store_true')
    p.add_argument('--save-json', action='store_true')
    p.add_argument('--project', default=ROOT / 'runs/val')
    p.add_argument('--name', default='exp')
    p.add_argument('--exist-ok', action='store_true')
    p.add_argument('--half', type=str2bool, default=True)
    p.add_argument('--pass_path', type=str, default=None)
    p.add_argument('--verbose', type=str2bool, default=False)
    p.add_argument('--card_bs1', type=str2bool, default=False)
    p.add_argument('--save_engine', type=str2bool, default=False)
    p.add_argument('--test-time', type=int, default=-1)
    p.add_argument('--model_name', type=str, default='yolov12n', choices=['yolov6n','yolov5','yolov12n'])
    opt = p.parse_args()
    opt.save_json |= opt.data_path.endswith('coco') or opt.data_path.endswith('coco/')
    if "coco_yolo_animal" in opt.data_path:
        opt.data_type = "coco_yolo_animal"
    print_args(vars(opt))
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    test_time = opt.test_time
    delattr(opt, 'test_time')
    if test_time > 0:
        print(f"Stress-Testing-pid:{os.getpid()}")
        st = time.time()
        while True:
            run(**vars(opt))
            if (time.time() - st)/3600 > test_time:
                break
    else:
        run(**vars(opt))

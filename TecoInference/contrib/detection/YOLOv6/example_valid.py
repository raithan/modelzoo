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
import os
import sys
import json
import time
import argparse
from tqdm import tqdm
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
import numpy as np
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from engine.tecoinfer_pytorch import TecoInferEngine

from utils.datasets.yolo_pt_dataset import create_dataloader
from utils.preprocess.pytorch.yolo_pt import preprocess
from utils.postprocess.pytorch.yolo_pt import (LOGGER, colorstr, increment_path, scale_boxes, xywh2xyxy,
        xyxy2xywh, print_args, box_iou, ap_per_class, postprocess, TQDM_BAR_FORMAT, CLASSES, CLASSES_ANIMAL)

MAX_ENGINE_NUMS = int(os.getenv('MAX_ENGINE_NUMS', 4))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_data(data_path):
    # check coco val data
    val = os.path.join(data_path, "val2017.txt")
    val_path = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
    if not all(x.exists() for x in val_path):
        LOGGER.info('\nDataset not found ⚠️, missing paths %s' % [str(x) for x in val if not x.exists()])
        raise Exception('Dataset not found ❌')
    return val

def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})

def metrics_per_batch(im, preds, targets, paths, shapes, seen, jdict, stats, save_json=False, single_cls=False,):
    # coco80_to_coco91_class
    class_map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 
                39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    # Metrics
    for si, pred in enumerate(preds):
        labels = targets[targets[:, 0] == si, 1:]
        nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
        path, shape = Path(paths[si]), shapes[si][0]
        correct = torch.zeros(npr, niou, dtype=torch.bool)  # init
        seen += 1

        if npr == 0:
            if nl:
                stats.append((correct, *torch.zeros((2, 0)), labels[:, 0]))
            continue

        # Predictions
        if single_cls:
            pred[:, 5] = 0
        predn = pred.clone()
        scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            correct = process_batch(predn, labelsn, iouv)
        stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

        # Save/log
        if save_json:
            save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary

def compute_metrics(ckpt, dataloader, stats, jdict, seen, data_path, save_dir, data_type, save_json=False,  plots=False,):

    class_type = {
        "coco_yolo_animal": CLASSES_ANIMAL,
        "coco": CLASSES,
    }
    names = dict(enumerate(class_type[data_type]))
    nc = len(class_type[data_type])
    ap, ap_class = [], []

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # pycocotools val
    if save_json and len(jdict):
        w = ckpt.split("/")[-1]
        anno_json = os.path.join(data_path, 'annotations/instances_val2017.json') # annotations json
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    print('eval_metric',map)
    return map

def run(model_name="yolov6n",
        ckpt="./yolov6n.onnx",
        input_name='images',
        target='sdaa',
        batch_size=64,
        shape=640,
        half=True,  # use FP16 half-precision inference
        data_path='',
        stride=32,  # default stride
        single_cls=False,
        pad=0.5,
        rect=False,
        workers=0,
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        save_json=False,  # save a COCO-JSON results file
        save_dir=Path(''),
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        verbose=False,
        save_engine=False,
        pass_path=None,
        data_type="coco",
        card_bs1=False,
        ):
    # build tvm-model
    input_size = [[max(batch_size // MAX_ENGINE_NUMS, 1), 3, shape, shape]]
    pipeline = TecoInferEngine(ckpt=ckpt,
                                input_name=input_name,
                                target=target,
                                model_name=model_name,
                                batch_size=batch_size,
                                input_size=input_size,
                                dtype="float16" if half else "float32",
                                save_engine=save_engine,
                                pass_path=pass_path,
                                card_bs1=card_bs1)
    # create dataloader
    val_path = check_data(data_path)
    dataloader = create_dataloader(val_path,
                                    shape,
                                    batch_size,
                                    stride,
                                    single_cls,
                                    pad=pad,
                                    rect=rect,
                                    workers=workers,
                                    prefix=colorstr(f'{"val"}: '))[0]

    # init parameters
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    seen = 0
    jdict, stats = [], []

    e2e_time, pre_time, run_time, post_time, ips = [], [], [], [], []
    max_step = int(os.environ.get("TECO_INFER_PIPELINES_MAX_STEPS", -1))
    warmup_step = int(os.environ.get("TECO_INFER_PIPELINES_WARMUP_STEPS", 0))
    global_step = 1

    while True:
        for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
            nb, _, height, width = im.shape  # batch size, channels, height, width
            start_time = time.time()

            # Preprocess
            dealed_im, padding_shape, _ = preprocess(im.numpy(), batch_size, (height, width), half=half)
            preprocess_time = time.time() - start_time

            # Inference
            preds = pipeline(dealed_im, conf_thres=conf_thres,
                                iou_thres=iou_thres,
                                max_det=max_det,
                                batch_padding=True,)

            if "coco_yolo_animal" in data_path:
                preds = preds[0]

            # NMS
            preds = postprocess(preds, padding_shape, conf_thres, iou_thres, labels=[], multi_label=True, max_det=max_det)
            infer_time = time.time() - start_time

            targets[:, 2:] *= torch.tensor((width, height, width, height))  # to pixels
            # Metrics
            metrics_per_batch(im, preds, targets, paths, shapes, seen, jdict, stats,
                                save_json=save_json, single_cls=single_cls,)

            if global_step > warmup_step:
                postprocess_time = infer_time - pipeline.run_time - preprocess_time
                sps = batch_size / infer_time
                e2e_time.append(infer_time)
                pre_time.append(preprocess_time)
                run_time.append(pipeline.run_time)
                post_time.append(postprocess_time)
                ips.append(sps)
                if verbose:
                    print(f'e2e_time: {infer_time}, inference_time: {pipeline.run_time}, preprocess_time: {preprocess_time}, postprocess: {postprocess_time}, sps: {sps}')
            
            if max_step > 0 and global_step == max_step:
                break
            global_step += 1
        if global_step >= max_step:
            break

    # 释放device显存，stream等资源
    if "sdaa" in target:
        pipeline.release()

    # Compute metrics
    _ = compute_metrics(ckpt, dataloader, stats, jdict, seen, data_path,
                save_dir, data_type, save_json=save_json, plots=False,)

    count = len(ips)
    print(f'summary: avg_sps: {sum(ips)/count} images/s, e2e_time: {sum(e2e_time)} s, avg_inference_time: {sum(run_time[5:])/(count-5)} s, avg_preprocess_time: {sum(pre_time)/count} s, avg_postprocess: {sum(post_time)/count} s')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./yolov6n.onnx', help='onnx path')
    parser.add_argument('--data-path', type=str, default='/data/teco-data/coco/', help='dataset path')
    parser.add_argument('--data-type', type=str, default='coco', help='dataset path')
    parser.add_argument('--input_name', type=str, default='images', help='input name')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--shape', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--target', default='sdaa', help='sdaa or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', type=str2bool, default=True, help='use FP16 half-precision inference')
    parser.add_argument('--pass_path', type=str, default=None, help='pass_path for tvm')
    parser.add_argument('--verbose', type=str2bool, default=False, help='print speed per batch')
    parser.add_argument('--card_bs1', type=str2bool, default=False, help='1 card inference for bs1')
    parser.add_argument('--save_engine', type=str2bool, default=False, help='save engine file when use trt')
    parser.add_argument('--test-time', type=int, default=-1, help='inference test time (h)')
    parser.add_argument('--model_name', type=str, default="yolov6n", help='model name for of yolo_pt',
                choices=["yolov6n", "yolov5"])
    opt = parser.parse_args()
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

        test_start_time = time.time()
        while True:
            run(**vars(opt))
            tested_time = (time.time() - test_start_time) / 60 / 60  # 已运行时间 (h)
            if tested_time > test_time:
                break
    else:
        run(**vars(opt))

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
import cv2
import random
import argparse
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from engine.tecoinfer_pytorch import TecoInferEngine

from utils.preprocess.pytorch.yolo_pt import preprocess, IMG_FORMATS
from utils.postprocess.pytorch.yolo_pt import postprocess


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

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="path_to/yolov6n.onnx", help='onnx path')
    parser.add_argument('--data-path', type=str, default='./imgs/coco/', help='images path')
    parser.add_argument('--data-type', type=str, default='coco', help='images path')
    parser.add_argument('--input_name', type=str, default='images', help='input name')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--shape', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--target', default='sdaa', help='sdaa or cpu')
    parser.add_argument('--half', type=str2bool, default=True, help='use FP16 half-precision inference')
    parser.add_argument('--pass_path', type=str, default=None, help='pass_path for tvm')
    parser.add_argument('--model_name', type=str, default="yolov6n", help='model name for of yolo_pt',
                choices=["yolov6n", "yolov5"])
    parser.add_argument('--save_result', type=bool, default=False, help='whether to save result image')
    opt = parser.parse_args()
    opt.dtype = "float16" if opt.half else "float32"
    opt.class_name=None

    return opt


def draw_boxes(opt,results):

    image_path = opt.data_path
    image = cv2.imread(image_path)

    def get_random_color():
        return [random.randint(0, 255) for _ in range(3)]
    # 绘制边界框
    for detection in results:
        for class_name, bounding_boxes in detection.items():
            color = get_random_color()
            for bbox in bounding_boxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    # 保存绘制后的图像
    output_path = 'output_image.jpg'
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    opt = parse_opt()

    input_size = [[max(opt.batch_size // MAX_ENGINE_NUMS, 1), 3, opt.shape, opt.shape]]
    pipeline = TecoInferEngine(ckpt=opt.ckpt,
                                input_name=opt.input_name,
                                target=opt.target,
                                model_name=opt.model_name,
                                batch_size=opt.batch_size,
                                input_size=input_size,
                                dtype=opt.dtype,
                                pass_path=opt.pass_path)

    for filename in os.listdir(opt.data_path):
        try:
            filepath = os.path.join(opt.data_path, filename)
            # 检查是否为图片
            if filepath.split('.')[-1].lower() not in IMG_FORMATS:
                continue
            im, padding_shape, image0_shapes = preprocess(filepath, opt.batch_size, (opt.shape, opt.shape), half=opt.dtype=='float16')
            results = pipeline(im)
            results = postprocess(results, padding_shape, image=im, image0_shapes=image0_shapes, class_name=opt.class_name)
            if len(results) == 0:
                print(f"{filename}: no detections")
            else:
                print(f"{filename}:")
                for i in range(len(results)):
                    for k,v in results[i].items():
                        print(k,v)
        except IOError:
            print("无法打开图片文件:", filepath)
    # 释放device显存，stream等资源
    if "sdaa" in opt.target:
        pipeline.release()





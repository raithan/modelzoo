#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

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

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO 全参数解析")

    # 任务类型和模式
    parser.add_argument('--task', type=str, default='detect', help="YOLO任务类型 detect/segment/classify/pose/obb")
    parser.add_argument('--mode', type=str, default='train', help="YOLO模式 train/val/predict/export/track/benchmark")

    # Train settings
    parser.add_argument('--model', type=str, help="模型文件路径，如 yolov8n.pt 或 yolov8n.yaml")
    parser.add_argument('--data', type=str, help="数据集配置文件路径，如 coco8.yaml")
    parser.add_argument('--epochs', type=int, default=100, help="训练轮数")
    parser.add_argument('--time', type=float, help="训练时长（小时），如指定则覆盖epochs")
    parser.add_argument('--patience', type=int, default=100, help="早停等待轮数")
    parser.add_argument('--batch', type=int, default=16, help="每批次图片数量")
    parser.add_argument('--imgsz', type=int, default=640, help="输入图片尺寸")
    parser.add_argument('--save', type=bool, default=True, help="是否保存训练模型与结果")
    parser.add_argument('--save_period', type=int, default=-1, help="每隔多少epoch保存一次，<1不保存")
    parser.add_argument('--cache', type=str, default='False', help="数据加载缓存方式 True/ram/disk/False")
    parser.add_argument('--device', type=str, default='', help="训练设备 'cpu'/'mps'/0/[0,1,2,3]/-1")
    parser.add_argument('--workers', type=int, default=8, help="数据加载线程数")
    parser.add_argument('--project', type=str, default='', help="项目名")
    parser.add_argument('--name', type=str, default='', help="实验名")
    parser.add_argument('--exist_ok', action='store_true', help="是否覆盖已有实验")
    parser.add_argument('--pretrained', type=str, default='True', help="是否用预训练模型")
    parser.add_argument('--optimizer', type=str, default='auto', choices=['SGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'], help="优化器类型")
    parser.add_argument('--verbose', type=bool, default=True, help="是否输出详细日志")
    parser.add_argument('--seed', type=int, default=0, help="随机种子")
    parser.add_argument('--deterministic', type=bool, default=True, help="是否开启确定性训练")
    parser.add_argument('--single_cls', type=bool, default=False, help="是否将多类别数据当作单类别训练")
    parser.add_argument('--rect', type=bool, default=False, help="是否使用矩形训练")
    parser.add_argument('--cos_lr', type=bool, default=False, help="是否使用余弦学习率")
    parser.add_argument('--close_mosaic', type=int, default=10, help="最后多少epoch关闭mosaic增强")
    parser.add_argument('--resume', type=bool, default=False, help="是否从断点恢复训练")
    parser.add_argument('--amp', type=bool, default=True, help="是否启用混合精度")
    parser.add_argument('--fraction', type=float, default=1.0, help="训练集使用比例")
    parser.add_argument('--profile', type=bool, default=False, help="训练时profile ONNX/TRT速度")
    parser.add_argument('--freeze', type=str, default='', help="冻结层数或层索引列表")
    parser.add_argument('--multi_scale', type=bool, default=False, help="是否使用多尺度训练")
    # Segmentation
    parser.add_argument('--overlap_mask', type=bool, default=True, help="是否合并mask（仅segment任务）")
    parser.add_argument('--mask_ratio', type=int, default=4, help="mask下采样比例（仅segment任务）")
    # Classification
    parser.add_argument('--dropout', type=float, default=0.0, help="分类任务dropout比率")

    # Val/Test settings
    parser.add_argument('--val', type=bool, default=True, help="训练时是否验证")
    parser.add_argument('--split', type=str, default='val', help="验证集split名称")
    parser.add_argument('--save_json', type=bool, default=False, help="是否保存结果为json")
    parser.add_argument('--conf', type=float, help="推理/验证的置信度阈值")
    parser.add_argument('--iou', type=float, default=0.7, help="NMS的IoU阈值")
    parser.add_argument('--max_det', type=int, default=300, help="单图最大检测数量")
    parser.add_argument('--half', type=bool, default=False, help="是否用FP16")
    parser.add_argument('--dnn', type=bool, default=False, help="ONNX推理是否用OpenCV DNN")
    parser.add_argument('--plots', type=bool, default=True, help="是否保存训练/验证过程图")

    # Predict settings
    parser.add_argument('--source', type=str, default='', help="预测图片/视频目录")
    parser.add_argument('--vid_stride', type=int, default=1, help="视频帧采样步长")
    parser.add_argument('--stream_buffer', type=bool, default=False, help="视频推理是否全部缓存")
    parser.add_argument('--visualize', type=bool, default=False, help="可视化模型特征")
    parser.add_argument('--augment', type=bool, default=False, help="预测时是否数据增强")
    parser.add_argument('--agnostic_nms', type=bool, default=False, help="类别无关NMS")
    parser.add_argument('--classes', type=str, default='', help="只保留哪些类别")
    parser.add_argument('--retina_masks', type=bool, default=False, help="高分辨率mask")
    parser.add_argument('--embed', type=str, default='', help="输出哪些层的embedding")

    # Visualize settings
    parser.add_argument('--show', type=bool, default=False, help="推理时是否显示图像")
    parser.add_argument('--save_frames', type=bool, default=False, help="是否保存视频逐帧结果")
    parser.add_argument('--save_txt', type=bool, default=False, help="是否保存txt结果")
    parser.add_argument('--save_conf', type=bool, default=False, help="是否保存置信度")
    parser.add_argument('--save_crop', type=bool, default=False, help="是否保存裁剪结果")
    parser.add_argument('--show_labels', type=bool, default=True, help="显示标签")
    parser.add_argument('--show_conf', type=bool, default=True, help="显示置信度")
    parser.add_argument('--show_boxes', type=bool, default=True, help="显示检测框")
    parser.add_argument('--line_width', type=int, help="检测框线宽")

    # Export settings
    parser.add_argument('--format', type=str, default='torchscript', help="导出格式")
    parser.add_argument('--keras', type=bool, default=False, help="是否导出Keras模型")
    parser.add_argument('--optimize', type=bool, default=False, help="TorchScript模型是否优化")
    parser.add_argument('--int8', type=bool, default=False, help="CoreML/TF是否INT8量化")
    parser.add_argument('--dynamic', type=bool, default=False, help="ONNX/TRT是否动态尺寸")
    parser.add_argument('--simplify', type=bool, default=True, help="ONNX模型是否简化")
    parser.add_argument('--opset', type=int, help="ONNX opset版本")
    parser.add_argument('--workspace', type=float, help="TensorRT workspace GiB")
    parser.add_argument('--nms', type=bool, default=False, help="CoreML是否加NMS")

    # Hyperparameters
    parser.add_argument('--lr0', type=float, default=0.01, help="初始学习率")
    parser.add_argument('--lrf', type=float, default=0.01, help="最终学习率(lr0*lrf)")
    parser.add_argument('--momentum', type=float, default=0.937, help="动量")
    parser.add_argument('--weight_decay', type=float, default=0.00001, help="权重衰减")
    parser.add_argument('--warmup_epochs', type=float, default=3.0, help="warmup轮数")
    parser.add_argument('--warmup_momentum', type=float, default=0.8, help="warmup动量")
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1, help="warmup bias lr")
    parser.add_argument('--box', type=float, default=7.5, help="box损失系数")
    parser.add_argument('--cls', type=float, default=0.5, help="cls损失系数")
    parser.add_argument('--dfl', type=float, default=1.5, help="dfl损失系数")
    parser.add_argument('--pose', type=float, default=12.0, help="pose损失系数")
    parser.add_argument('--kobj', type=float, default=1.0, help="关键点obj损失系数")
    parser.add_argument('--nbs', type=int, default=64, help="名义batch size")
    parser.add_argument('--hsv_h', type=float, default=0.015, help="HSV-Hue增强")
    parser.add_argument('--hsv_s', type=float, default=0.7, help="HSV-Sat增强")
    parser.add_argument('--hsv_v', type=float, default=0.4, help="HSV-Value增强")
    parser.add_argument('--degrees', type=float, default=0.0, help="图片旋转")
    parser.add_argument('--translate', type=float, default=0.1, help="图片平移")
    parser.add_argument('--scale', type=float, default=0.5, help="图片缩放")
    parser.add_argument('--shear', type=float, default=0.0, help="图片剪切")
    parser.add_argument('--perspective', type=float, default=0.0, help="图片透视")
    parser.add_argument('--flipud', type=float, default=0.0, help="上下翻转概率")
    parser.add_argument('--fliplr', type=float, default=0.5, help="左右翻转概率")
    parser.add_argument('--bgr', type=float, default=0.0, help="BGR通道扰动概率")
    parser.add_argument('--mosaic', type=float, default=0.0, help="mosaic增强概率")
    parser.add_argument('--mixup', type=float, default=0.0, help="mixup概率")
    parser.add_argument('--cutmix', type=float, default=0.0, help="cutmix概率")
    parser.add_argument('--copy_paste', type=float, default=0.0, help="copy-paste概率")
    parser.add_argument('--copy_paste_mode', type=str, default="flip", help="copy-paste方式")
    parser.add_argument('--auto_augment', type=str, default="randaugment", help="auto augmentation policy")
    parser.add_argument('--erasing', type=float, default=0.4, help="分类任务随机擦除概率")

    # Custom config
    parser.add_argument('--cfg', type=str, default='', help="自定义config.yaml路径")

    # Tracker settings
    parser.add_argument('--tracker', type=str, default='botsort.yaml', help="目标跟踪tracker")

    args = parser.parse_args()
    return args

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
"""Centralized catalog of paths."""
import os


class DatasetCatalog(object):
    DATA_DIR = '/data/teco-data' # 总路径 修改为当前的
    DATASETS = {
        'coco_2017_train': {
            'img_dir': 'COCO/train2017',
            'ann_file': 'COCO/annotations/instances_train2017.json'
        },
        'coco_2017_val': {
            'img_dir': 'COCO/val2017',
            'ann_file': 'COCO/annotations/instances_val2017.json'
        },
        'coco_2017_test_dev': {
            'img_dir': 'COCO/test2017',
            'ann_file': 'COCO/annotations/image_info_test-dev2017.json'
        },
        }

    @staticmethod
    def get(name):
        if 'coco' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs['img_dir']),
                ann_file=os.path.join(data_dir, attrs['ann_file']),
            )
            return dict(
                factory='COCODataset',
                args=args,
            )
        else:
            raise RuntimeError('Only support coco format now!')
        return None

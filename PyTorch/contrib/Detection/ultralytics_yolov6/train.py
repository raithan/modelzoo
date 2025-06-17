from ultralytics import YOLO
from torch_sdaa.utils import cuda_migrate  # 使用torch_sdaa自动迁移方法


if __name__ == '__main__':
    model = YOLO('/data/bigc-data/wyc/AAmodelzoo/ultralytics/ultralytics/cfg/models/v6/yolov6.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='/data/bigc-data/wyc/AAmodelzoo/ultralytics/ultralytics/cfg/datasets/coco128.yaml',
                cache=False,
                imgsz=640,
                epochs=25,
                batch=32,
                close_mosaic=0, # 最后多少个epoch关闭mosaic数据增强，设置0代表全程开启mosaic训练
                workers=0, 
                # device='[0,1]',
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp | loss出现nan可以关闭amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
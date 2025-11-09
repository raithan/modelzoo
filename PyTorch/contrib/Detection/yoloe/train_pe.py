from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train_pe import YOLOEPETrainer, YOLOEPESegTrainer
import os
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import yaml_load, LOGGER
import torch

os.environ["PYTHONHASHSEED"] = "0"

data = "ultralytics/cfg/datasets/coco.yaml"

model_path = "yoloe-v8.yaml"

scale = guess_model_scale(model_path)
cfg_dir = "ultralytics/cfg"
default_cfg_path = f"{cfg_dir}/default.yaml"
# 请查看cfg_dir下的文件夹前缀名修改{scale}
# extend_cfg_path = f"{cfg_dir}/coco_{scale}_train.yaml"
extend_cfg_path = f"{cfg_dir}/coco_l_train.yaml"
defaults = yaml_load(default_cfg_path)
extends = yaml_load(extend_cfg_path)
assert(all(k in defaults for k in extends))
LOGGER.info(f"Extends: {extends}")

model = YOLOE(model_path)
with torch.inference_mode():
    model.model.eval()
    names = list(yaml_load(data)['names'].values())
    tpe = model.get_text_pe(names)

pe_path = "coco-pe.pt"
torch.save({"names": names, "pe": tpe}, pe_path)

head_index = len(model.model.model) - 1
freeze = [str(f) for f in range(0, head_index)]
for name, child in model.model.model[-1].named_children():
    if 'cv3' not in name:
        freeze.append(f"{head_index}.{name}")
freeze.extend([
    f"{head_index}.cv3.0.0", f"{head_index}.cv3.0.1",
    f"{head_index}.cv3.1.0", f"{head_index}.cv3.1.1",
    f"{head_index}.cv3.2.0", f"{head_index}.cv3.2.1"
])
  
model.train(data=data, epochs=51, close_mosaic=5, batch=16, 
            optimizer='AdamW', lr0=1e-3, warmup_bias_lr=0.0, \
            weight_decay=0.025, momentum=0.9, workers=2, \
            # 使用多个GPU时改为：device="0,1,2,3,4,5,6,7"
            device="0", **extends, \
            trainer=YOLOEPETrainer, freeze=freeze, train_pe_path=pe_path)
## 📌 1. 配置文件参数

| 参数               | 类型  | 默认值 | 说明                    |
| ---------------- | --- | --- | --------------------- |
| `-c`, `--config` | str | ""  | 指定 YAML 配置文件，用于覆盖默认参数 |

---

## 📁 2. Dataset 参数

| 参数                            | 类型   | 默认值          | 说明                          |
| ----------------------------- | ---- | ------------ | --------------------------- |
| `data`                        | str  | None         | 数据集路径（位置参数，已废弃）             |
| `--data-dir`                  | str  | None         | 数据集根目录                      |
| `--dataset`                   | str  | ""           | 数据集类型与名称，如 `torch/imagenet` |
| `--train-split`               | str  | "train"      | 训练集划分名                      |
| `--val-split`                 | str  | "validation" | 验证集划分名                      |
| `--train-num-samples`         | int  | None         | IterableDataset 手动指定训练样本数   |
| `--val-num-samples`           | int  | None         | IterableDataset 手动指定验证样本数   |
| `--dataset-download`          | flag | False        | 是否自动下载数据集                   |
| `--class-map`                 | str  | ""           | 指定 class-to-index 映射文件      |
| `--input-img-mode`            | str  | None         | 输入图像转换模式，例如 RGB/L 模式        |
| `--input-key`                 | str  | None         | 数据集中输入图像字段名称                |
| `--target-key`                | str  | None         | 数据集中标签字段名称                  |
| `--dataset-trust-remote-code` | flag | False        | 允许 HF dataset 执行远程代码        |

---

## 🧠 3. Model 参数

| 参数                               | 类型     | 默认值        | 说明                       |
| -------------------------------- | ------ | ---------- | ------------------------ |
| `--model`                        | str    | "resnet50" | 模型名称                     |
| `--pretrained`                   | flag   | False      | 是否使用预训练模型                |
| `--pretrained-path`              | str    | None       | 将指定 checkpoint 作为预训练权重加载 |
| `--initial-checkpoint`           | str    | ""         | 初始化后加载的 checkpoint       |
| `--resume`                       | str    | ""         | 从 checkpoint 恢复训练        |
| `--no-resume-opt`                | flag   | False      | 恢复训练时不加载优化器状态            |
| `--num-classes`                  | int    | None       | 分类类别数                    |
| `--gp`                           | str    | None       | 全局池化方式（avg, max, fast 等） |
| `--img-size`                     | int    | None       | 输入尺寸（正方形）                |
| `--in-chans`                     | int    | None       | 输入通道数                    |
| `--input-size`                   | 3 ints | None       | 输入尺寸 (C H W)             |
| `--crop-pct`                     | float  | None       | 验证集中心裁剪比例                |
| `--mean`                         | list   | None       | 数据集均值                    |
| `--std`                          | list   | None       | 数据集标准差                   |
| `--interpolation`                | str    | ""         | 图像插值方式                   |
| `-b`, `--batch-size`             | int    | 128        | 训练 batch 大小              |
| `-vb`, `--validation-batch-size` | int    | None       | 验证 batch 大小              |
| `--channels-last`                | flag   | False      | 使用 NHWC 内存布局             |
| `--fuser`                        | str    | ""         | TorchScript fuser 类型     |
| `--grad-accum-steps`             | int    | 1          | 梯度累积步数                   |
| `--grad-checkpointing`           | flag   | False      | 开启梯度检查点                  |
| `--fast-norm`                    | flag   | False      | 使用实验性快速 norm             |
| `--model-kwargs`                 | KV     | {}         | 传入模型构造函数的 kwargs         |
| `--head-init-scale`              | float  | None       | 头层 scale 初始化             |
| `--head-init-bias`               | float  | None       | 头层 bias 初始化              |
| `--torchcompile-mode`            | str    | None       | torch.compile 编译模式       |

### Scripting / Codegen

| 参数               | 类型   | 默认值   | 说明                            |
| ---------------- | ---- | ----- | ----------------------------- |
| `--torchscript`  | flag | False | 使用 torch.jit.script           |
| `--torchcompile` | str  | None  | torch.compile 后端（默认 inductor） |

---

## 🖥️ 4. Device & Distributed 参数

| 参数                   | 类型   | 默认值       | 说明                     |
| -------------------- | ---- | --------- | ---------------------- |
| `--device`           | str  | "cuda"    | 使用设备类型                 |
| `--amp`              | flag | False     | 开启混合精度训练               |
| `--amp-dtype`        | str  | "float16" | AMP dtype（fp16 / bf16） |
| `--amp-impl`         | str  | "native"  | AMP 实现（native / apex）  |
| `--model-dtype`      | str  | None      | 模型数据类型                 |
| `--no-ddp-bb`        | flag | False     | DDP 不广播 buffers        |
| `--synchronize-step` | flag | False     | 每 step 手动 sync         |
| `--local_rank`       | int  | 0         | DDP local rank         |
| `--device-modules`   | list | None      | 自定义设备模块                |

---

## ⚙️ 5. 优化器参数 (Optimizer)

| 参数                           | 类型       | 默认值    | 说明                   |
| ---------------------------- | -------- | ------ | -------------------- |
| `--opt`                      | str      | "sgd"  | 优化器类型                |
| `--opt-eps`                  | float    | None   | epsilon              |
| `--opt-betas`                | 2 floats | None   | Adam 类优化器的 betas     |
| `--momentum`                 | float    | 0.9    | SGD 动量               |
| `--weight-decay`             | float    | 2e-5   | 权重衰减                 |
| `--clip-grad`                | float    | None   | 梯度裁剪阈值               |
| `--clip-mode`                | str      | "norm" | 裁剪方式（norm/value/agc） |
| `--layer-decay`              | float    | None   | 层级 lr decay          |
| `--layer-decay-min-scale`    | float    | 0      | decay 最小比例           |
| `--layer-decay-no-opt-scale` | float    | None   | 禁止优化缩放的层             |
| `--opt-kwargs`               | KV       | {}     | 优化器扩展参数              |

---

## 📈 6. Learning Rate Schedule 参数

| 参数                   | 类型        | 默认值          | 说明                      |
| -------------------- | --------- | ------------ | ----------------------- |
| `--sched`            | str       | "cosine"     | LR scheduler 类型         |
| `--sched-on-updates` | flag      | False        | 每 step 而不是每 epoch 更新 LR |
| `--lr`               | float     | None         | 手动 lr（优先级最高）            |
| `--lr-base`          | float     | 0.1          | 基础 lr（自动缩放）             |
| `--lr-base-size`     | int       | 256          | 基准 batch size           |
| `--lr-base-scale`    | str       | ""           | lr 缩放方式（linear/sqrt）    |
| `--lr-noise`         | list      | None         | lr 扰动百分比范围              |
| `--lr-noise-pct`     | float     | 0.67         | 扰动比例                    |
| `--lr-noise-std`     | float     | 1.0          | 扰动标准差                   |
| `--lr-cycle-mul`     | float     | 1.0          | cycle 倍数                |
| `--lr-cycle-decay`   | float     | 0.5          | cycle 衰减                |
| `--lr-cycle-limit`   | int       | 1            | cycle 次数                |
| `--lr-k-decay`       | float     | 1.0          | cosine/poly k-decay     |
| `--warmup-lr`        | float     | 1e-5         | warmup 初始 lr            |
| `--min-lr`           | float     | 0            | 最小 lr                   |
| `--epochs`           | int       | 300          | 训练 epoch 数              |
| `--epoch-repeats`    | float     | 0            | 一个 epoch 重复几次           |
| `--start-epoch`      | int       | None         | 手动设置起始 epoch            |
| `--decay-milestones` | list[int] | [90,180,270] | MultiStepLR 的里程碑        |
| `--decay-epochs`     | float     | 90           | 周期性 decay 间隔            |
| `--warmup-epochs`    | int       | 5            | warmup epoch 数          |
| `--warmup-prefix`    | flag      | False        | warmup 不计入 decay        |
| `--cooldown-epochs`  | int       | 0            | scheduler 冷却 epoch      |
| `--patience-epochs`  | int       | 10           | Plateau patience        |
| `--decay-rate`       | float     | 0.1          | lr 衰减率                  |

---

## 🎨 7. Augmentation & Regularization 参数

| 参数                      | 类型       | 默认值         | 说明                       |
| ----------------------- | -------- | ----------- | ------------------------ |
| `--no-aug`              | flag     | False       | 禁用所有增强                   |
| `--train-crop-mode`     | str      | None        | 训练裁剪模式                   |
| `--scale`               | 2 floats | [0.08,1.0]  | 随机裁剪比例范围                 |
| `--ratio`               | 2 floats | [0.75,1.33] | 随机裁剪宽高比范围                |
| `--hflip`               | float    | 0.5         | 水平翻转概率                   |
| `--vflip`               | float    | 0.0         | 垂直翻转概率                   |
| `--color-jitter`        | float    | 0.4         | 颜色抖动强度                   |
| `--color-jitter-prob`   | float    | None        | 颜色抖动概率                   |
| `--grayscale-prob`      | float    | None        | 灰度化概率                    |
| `--gaussian-blur-prob`  | float    | None        | 高斯模糊概率                   |
| `--aa`                  | str      | None        | AutoAugment 策略           |
| `--aug-repeats`         | float    | 0           | 数据增强重复次数                 |
| `--aug-splits`          | int      | 0           | 增强分支数                    |
| `--jsd-loss`            | flag     | False       | 开启 JSD loss              |
| `--bce-loss`            | flag     | False       | Mixup/Cutmix + BCE       |
| `--bce-sum`             | flag     | False       | BCE loss 在类别维上求和         |
| `--bce-target-thresh`   | float    | None        | BCE 软标签二值化阈值             |
| `--bce-pos-weight`      | float    | None        | BCE 正样本权重                |
| `--reprob`              | float    | 0           | RandomErase 概率           |
| `--remode`              | str      | "pixel"     | RandomErase 模式           |
| `--recount`             | int      | 1           | RandomErase 次数           |
| `--resplit`             | flag     | False       | 不让 erase 作用于 clean split |
| `--mixup`               | float    | 0           | mixup α                  |
| `--cutmix`              | float    | 0           | cutmix α                 |
| `--cutmix-minmax`       | list     | None        | min/max cutmix 区间        |
| `--mixup-prob`          | float    | 1.0         | mixup/cutmix 概率          |
| `--mixup-switch-prob`   | float    | 0.5         | mixup → cutmix 切换概率      |
| `--mixup-mode`          | str      | "batch"     | batch/pair/elem          |
| `--mixup-off-epoch`     | int      | 0           | 关闭 mixup 的 epoch         |
| `--smoothing`           | float    | 0.1         | label smoothing          |
| `--train-interpolation` | str      | "random"    | 训练插值方式                   |
| `--drop`                | float    | 0           | dropout                  |


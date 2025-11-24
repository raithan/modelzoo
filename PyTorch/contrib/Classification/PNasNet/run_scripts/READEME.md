## 参数介绍
参数名 | 解释 | 样例
-----------------|-----------------|-----------------
`--data-dir` | ImageNet 数据集根目录路径。 | `--data-dir /data/teco-data/imagenet/`
`--model` | 模型名称。 | `--model pnasnet5large`
`-b, --batch-size` | 训练 batch size。 | `-b 128`
`--sched` | 学习率 scheduler 类型。 | `--sched step`
`--epochs` | 训练轮数。 | `--epochs 450`
`--decay-epochs` | 学习率衰减周期（step scheduler 间隔）。 | `--decay-epochs 2.4`
`--decay-rate` | 学习率衰减系数。 | `--decay-rate .97`
`--opt` | 优化器类型。 | `--opt rmsproptf`
`--opt-eps` | 优化器 eps。 | `--opt-eps .001`
`-j, --workers` | DataLoader 线程数。 | `-j 8`
`--warmup-lr` | warmup 初始学习率。 | `--warmup-lr 1e-6`
`--weight-decay` | 权重衰减。 | `--weight-decay 1e-5`
`--drop` | Dropout 概率。 | `--drop 0.3`
`--drop-path` | DropPath 概率。 | `--drop-path 0.2`
`--model-ema` | 是否启用 EMA 模型。 | `--model-ema`
`--model-ema-decay` | EMA 衰减因子。 | `--model-ema-decay 0.9999`
`--aa` | AutoAugment 策略。 | `--aa rand-m9-mstd0.5`
`--remode` | Random Erase 模式。 | `--remode pixel`
`--reprob` | Random Erase 概率。 | `--reprob 0.2`
`--amp` | 是否启用混合精度训练。 | `--amp`
`--lr` | 学习率（显式指定，覆盖根据 batch_size 自动计算的 lr）。 | `--lr .016`

---
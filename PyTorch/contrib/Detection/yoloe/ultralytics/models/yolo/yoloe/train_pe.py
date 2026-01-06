from ultralytics.models.yolo.yoloe.train_yoloe import YOLOETrainerFromScratch, YOLOETrainer
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer
from copy import deepcopy
import torch
from ultralytics.models.yolo.detect import DetectionValidator
from copy import copy
from ultralytics.nn.tasks import YOLOEModel, YOLOESegModel
from ultralytics.utils import DEFAULT_CFG, RANK

# =============== Add: per-iteration logging callbacks ===============
import time
import datetime
from ultralytics.utils import callbacks as yolo_callbacks


def _tcappdll_on_train_epoch_start(trainer):
    trainer._tcappdll_prev_iter_t = time.time()
    trainer._tcappdll_iter = -1


def _tcappdll_on_train_batch_start(trainer):
    trainer._tcappdll_iter_t0 = time.time()


def _get_loss_value(trainer):
    try:
        if hasattr(trainer, "loss") and trainer.loss is not None:
            return float(trainer.loss.detach().item())
    except Exception:
        pass
    try:
        if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
            return float(sum(trainer.loss_items))
    except Exception:
        pass
    return 0.0


def _tcappdll_on_train_batch_end(trainer):
    if RANK not in (-1, 0):
        return

    trainer._tcappdll_iter = getattr(trainer, "_tcappdll_iter", -1) + 1
    now = time.time()
    it_time = now - getattr(trainer, "_tcappdll_iter_t0", now)

    bs = getattr(trainer, "batch_size", None) or getattr(trainer.args, "batch", None) or 0
    ips = (bs / it_time) if it_time > 0 and bs else 0.0

    total_loss = _get_loss_value(trainer)

    # 打印分项 + 每目标平均
    lbox = lcls = ldfl = None
    if hasattr(trainer, "loss_items") and trainer.loss_items is not None and len(trainer.loss_items) >= 3:
        lbox, lcls, ldfl = [float(x) for x in trainer.loss_items[:3]]
        total_calc = lbox + lcls + ldfl
    else:
        total_calc = total_loss

    num_targets = getattr(trainer, "_last_num_targets", None)
    per_target = (total_calc / max(num_targets, 1)) if isinstance(num_targets, int) else None

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    epoch = getattr(trainer, "epoch", 0)
    it = trainer._tcappdll_iter
    rank_print = 0 if RANK == -1 else RANK

    extra = ""
    if lbox is not None:
        extra = f" box={lbox:.4f} cls={lcls:.4f} dfl={ldfl:.4f} total={total_calc:.4f}"
    if per_target is not None:
        extra += f" per_target={per_target:.6f} targets={num_targets}"
    print('/n')
    print(
        f"TCAPPDLL {ts} - Epoch:{epoch} Iteration:{it} rank:{rank_print} "
        f"train.loss_avg:{total_calc:.4f} train.ips:{ips:.2f} imgs/s train.time:{it_time:.4f}{extra}",
        flush=True,
    )


# 注册回调
try:
    yolo_callbacks.add_integration_callbacks()
except TypeError:
    pass
except Exception:
    pass

for event, fn in {
    "on_train_epoch_start": _tcappdll_on_train_epoch_start,
    "on_train_batch_start": _tcappdll_on_train_batch_start,
    "on_train_batch_end": _tcappdll_on_train_batch_end,
}.items():
    if hasattr(yolo_callbacks, "default_callbacks") and isinstance(yolo_callbacks.default_callbacks, dict):
        yolo_callbacks.default_callbacks.setdefault(event, []).append(fn)
# ======================= End of additions ===========================


class YOLOEPETrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return YOLOEModel initialized with specified config and weights."""
        model = YOLOEModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=3,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )

        # 移除保存的 PE 占位
        if hasattr(model.model[-1], "savpe"):
            del model.model[-1].savpe

        if weights:
            model.load(weights)

        # 加载并融合文本 PE（注意：fuse 内部可能在 inference_mode 下新建权重）
        model.eval()
        pe_state = torch.load(self.args.train_pe_path, map_location="cpu")
        model.set_classes(pe_state["names"], pe_state["pe"])
        model.model[-1].fuse(model.pe)

        # 关键修复：用 deepcopy 替换每个尺度 cls 分支的最后一层模块，新的参数不带 inference 标记
        for i in range(len(model.model[-1].cv3)):  # 通常 3 个尺度
            if len(model.model[-1].cv3[i]) > 2:
                old = model.model[-1].cv3[i][2]
                new = deepcopy(old)
                for p in new.parameters():
                    p.requires_grad = True
                model.model[-1].cv3[i][2] = new

        # 清理临时 PE
        if hasattr(model, "pe"):
            del model.pe
        model.train()

        return model

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        self._last_num_targets = int(batch["cls"].shape[0]) if "cls" in batch else 0
        return batch


class YOLOEPESegTrainer(SegmentationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return YOLOESegModel initialized with specified config and weights."""
        model = YOLOESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=3,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )

        if hasattr(model.model[-1], "savpe"):
            del model.model[-1].savpe

        if weights:
            model.load(weights)

        model.eval()
        pe_state = torch.load(self.args.train_pe_path, map_location="cpu")
        model.set_classes(pe_state["names"], pe_state["pe"])
        model.model[-1].fuse(model.pe)

        for i in range(len(model.model[-1].cv3)):
            if len(model.model[-1].cv3[i]) > 2:
                old = model.model[-1].cv3[i][2]
                new = deepcopy(old)
                for p in new.parameters():
                    p.requires_grad = True
                model.model[-1].cv3[i][2] = new

        if hasattr(model, "pe"):
            del model.pe
        model.train()
        return model

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        self._last_num_targets = int(batch["cls"].shape[0]) if "cls" in batch else 0
        return batch


class YOLOEPEFreeTrainer(YOLOEPETrainer, YOLOETrainerFromScratch):
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = ("box", "cls", "dfl")
        return DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        self._last_num_targets = int(batch["cls"].shape[0]) if "cls" in batch else 0
        return batch
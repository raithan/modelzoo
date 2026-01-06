#!/usr/bin/env python3 -u
# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# Apache 2.0 license.

"""
Train a new model on one or across multiple GPUs.
"""
import nltk
import os

# 设置 NLTK 数据路径
nltk_data_path = '/data/teco-data/ofa/dataset/nltk_data'
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

import argparse
import logging
import math
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable
import datetime
import time

# ------------------- 自定义毫秒 Formatter -------------------
class MsFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        msec = int(record.msecs)
        return f"{t},{msec:03d}"

# 先屏蔽 root logger（可选，保证干净）
root_logger = logging.getLogger()
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)
root_logger.propagate = False
root_logger.disabled = True  # 如需 root 输出，去掉这一行

# 初始化 base_task logger
logger = logging.getLogger("base_task")
logger.setLevel(logging.INFO)
# 清空已存在的 handlers（避免重复）
for h in list(logger.handlers):
    logger.removeHandler(h)
# 添加单一 StreamHandler
stream_handler = logging.StreamHandler(sys.stdout)
fmt = "INFO - %(asctime)s - base_task - TCAPPDLL %(message)s"
stream_handler.setFormatter(MsFormatter(fmt))
logger.addHandler(stream_handler)
# 阻止向上层传播，避免重复行
logger.propagate = False
# ------------------------------------------------------------

import numpy as np
import torch
from fairseq import (
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from omegaconf import DictConfig, OmegaConf

from utils import checkpoint_utils
from trainer import Trainer

# -------------------- 分布式安全封装 --------------------
def safe_dist_is_initialized() -> bool:
    try:
        return torch.distributed.is_available() and torch.distributed.is_initialized()
    except Exception:
        return False

def safe_get_rank() -> int:
    if safe_dist_is_initialized():
        return torch.distributed.get_rank()
    return 0

def safe_get_world_size() -> int:
    if safe_dist_is_initialized():
        return torch.distributed.get_world_size()
    return 1
# -------------------------------------------------------


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if distributed_utils.is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    # 若需要文件日志，可取消注释以下三行
    # if cfg.common.log_file is not None:
    #     file_handler = logging.FileHandler(filename=cfg.common.log_file)
    #     file_handler.setFormatter(MsFormatter("INFO - %(asctime)s - base_task - TCAPPDLL %(message)s"))
    #     logger.addHandler(file_handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    logger.info(str(cfg))

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logger.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    task = tasks.setup_task(cfg.task)
    assert cfg.criterion, "Please specify criterion to train a model"

    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)

    # bitfit
    if cfg.model.bitfit:
        for name, param in model.named_parameters():
            if ("layer_norm" in name and "bias" in name) or ("fc" in name and "bias" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

    criterion = task.build_criterion(cfg.criterion)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad)
        )
    )
    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad),
        )
    )

    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
    else:
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)

    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        disable_iterator_cache=True,
    )
    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm
        xm.rendezvous("load_checkpoint")

    max_epoch = cfg.optimization.max_epoch or math.inf
    if max_epoch > 0 and max_epoch != math.inf:
        total_num_updates = sum(
            math.ceil(len(epoch_itr) / cfg.optimization.update_freq[i])
            if i < len(cfg.optimization.update_freq) else
            math.ceil(len(epoch_itr) / cfg.optimization.update_freq[-1])
            for i in range(max_epoch)
        )
        trainer.lr_reinit(total_num_updates, trainer.get_num_updates())
    lr = trainer.get_lr()

    train_meter = meters.StopwatchMeter()
    train_meter.start()
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) <= --stop-min-lr={cfg.optimization.stop_min_lr}"
            )
            break

        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break

        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            load_dataset=True,
            disable_iterator_cache=True,
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info("ioPath PathManager waiting for all asynchronous checkpoint writes to finish.")
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False

    rank = safe_get_rank()
    world_size = safe_get_world_size()

    epoch_start_time = time.time()
    prev_iter_time = epoch_start_time
    if rank == 0:
        logger.info(f"Epoch {epoch_itr.epoch} start - Start iterating over samples")

    def _extract_batch_size(samples: Any) -> int:
        if isinstance(samples, list):
            total = 0
            for s in samples:
                if isinstance(s, dict):
                    if "nsentences" in s:
                        total += s["nsentences"]
                    elif "id" in s and hasattr(s["id"], "size"):
                        total += s["id"].size(0)
                    else:
                        ni = s.get("net_input", {})
                        for v in ni.values():
                            if torch.is_tensor(v) and v.size(0) > 0:
                                total += v.size(0)
                                break
                else:
                    try:
                        total += len(s)
                    except Exception:
                        pass
            return total if total > 0 else 1
        elif isinstance(samples, dict):
            if "nsentences" in samples:
                return samples["nsentences"]
            elif "id" in samples and hasattr(samples["id"], "size"):
                return samples["id"].size(0)
            else:
                ni = samples.get("net_input", {})
                for v in ni.values():
                    if torch.is_tensor(v) and v.size(0) > 0:
                        return v.size(0)
            return 1
        else:
            try:
                return len(samples)
            except Exception:
                return 1

    for i, samples in enumerate(itr):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            f"train_step-{i}"
        ):
            log_output = trainer.train_step(samples)

        iter_end_time = time.time()
        step_time = iter_end_time - prev_iter_time
        total_time = iter_end_time - epoch_start_time
        prev_iter_time = iter_end_time

        if log_output is not None:
            # 取 loss
            stats_inner = metrics.get_smoothed_values("train_inner")
            if "loss" in log_output:
                if "sample_size" in log_output and log_output["sample_size"] > 0:
                    raw_loss = log_output["loss"] / log_output["sample_size"]
                else:
                    raw_loss = log_output["loss"]
            elif "loss" in stats_inner:
                raw_loss = stats_inner["loss"]
            else:
                raw_loss = 0.0

            batch_size = _extract_batch_size(samples)
            ips = batch_size * world_size / (step_time if step_time > 0 else 1e-9)

            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            if rank == 0:
                logger.info(
                    f"{now_str} - Epoch: {epoch_itr.epoch} Iteration: {i+1} "
                    f"rank : {rank} train.loss : {raw_loss:.9f} "
                    f"train.ips : {ips:.2f} imgs/s train.total_time : {total_time:.6f} "
                    f"step_time : {step_time:.6f}"
                )

            metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )
        if should_stop:
            break

    if rank == 0:
        logger.info(f"end of epoch {epoch_itr.epoch} (average epoch stats below)")
    stats_epoch = metrics.get_smoothed_values("train")
    if rank == 0:
        logger.info(
            "Epoch {} summary: {}".format(
                epoch_itr.epoch,
                ", ".join(f"{k}={v}" for k, v in stats_epoch.items())
            )
        )
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours}"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)
        or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
        or should_stop
        or (
            cfg.dataset.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.dataset.validate_interval_updates == 0
        )
    ) and not cfg.dataset.disable_validation and num_updates >= cfg.dataset.validate_after_updates

    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )

    return valid_losses, should_stop


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    if cfg.dataset.fixed_validation_seed is not None:
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        logger.info(f'begin validation on "{subset}" subset')

        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
                    break
                trainer.valid_step(sample)

        stats = agg.get_smoothed_values()
        stats["num_updates"] = trainer.get_num_updates()
        progress.print(stats, tag=subset, step=trainer.get_num_updates())
        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}")

    if args.profile:
        with torch.sdaa.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cli_main()
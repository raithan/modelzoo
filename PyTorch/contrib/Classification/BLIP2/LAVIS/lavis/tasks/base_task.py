# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import time
from collections import deque
from datetime import datetime

import torch
import torch_sdaa
import torch.distributed as dist

from lavis.common.dist_utils import (
    get_rank,
    get_world_size,
    is_main_process,
    is_dist_avail_and_initialized,
)
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()
        self.inst_id_key = "instance_id"
        # 训练起始时间（用于 total_time）
        self._tcap_start_time = None
        self._tcap_last_iter_time = None

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        datasets = {}
        datasets_config = cfg.datasets_cfg
        assert len(datasets_config) > 0, "At least one dataset has to be specified."
        for name in datasets_config:
            dataset_config = datasets_config[name]
            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()
            datasets[name] = dataset
        return datasets

    def train_step(self, model, samples):
        """
        模型前向输出中提取所有包含 'loss' 的键，主损失约定为 output['loss']。
        """
        output = model(samples)
        loss_dict = {}
        for k, v in output.items():
            if "loss" in k:
                loss_dict[k] = v
        return output["loss"], loss_dict

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_training(self, model, dataset, **kwargs):
        model.before_training(dataset=dataset, task_type=type(self))

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, sdaa_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        print_freq = 10
        results = []
        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, sdaa_enabled=sdaa_enabled)
            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)
        if is_dist_avail_and_initialized():
            dist.barrier()
        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        sdaa_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
        extra_log_freq=None,    # 保留兼容
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(data_loader),
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            sdaa_enabled=sdaa_enabled,
            accum_grad_iters=accum_grad_iters,
            extra_log_freq=extra_log_freq,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        sdaa_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
        extra_log_freq=None,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            sdaa_enabled=sdaa_enabled,
            accum_grad_iters=accum_grad_iters,
            extra_log_freq=extra_log_freq,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        sdaa_enabled=False,
        accum_grad_iters=1,
        extra_log_freq=None,   # 不再使用窗口统计，但保留参数兼容
    ):
        """
        每一次迭代输出 TCAPPDLL 行：
        TCAPPDLL <timestamp> - Epoch: E Iteration: G rank : R train.loss : L train.ips : IPS imgs/s train.total_time : T
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        # 初始化 metric logger（仍用于最终平均和 lr/log.txt）
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(f"Start training epoch {epoch}, {iters_per_epoch} iters per inner epoch.")

        header = f"Train: data epoch: [{epoch}]"
        if start_iters is None:
            inner_epoch = epoch
            global_iter_base = 0
        else:
            inner_epoch = start_iters // iters_per_epoch
            header += f"; inner epoch [{inner_epoch}]"
            global_iter_base = start_iters

        # TCAP 时间基准
        if self._tcap_start_time is None:
            self._tcap_start_time = time.time()
            self._tcap_last_iter_time = self._tcap_start_time

        output_dir = registry.get_path("output_dir")
        tcap_log_path = os.path.join(output_dir, "tcap.log")
        if is_main_process() and (not os.path.exists(tcap_log_path)):
            with open(tcap_log_path, "w") as f:
                f.write("# TCAP Style Iter Log\n")
                f.write("# Format: TCAPPDLL YYYY-MM-DD HH:MM:SS.ffffff - Epoch: E Iteration: G rank : R train.loss : L train.ips : IPS imgs/s train.total_time : T step_time : S\n\n")

        # 使用 log_freq 控制原有 metric_logger 行，TCAP 始终每步输出
        if log_freq is None or log_freq <= 0:
            log_freq = 10**12  # 实际上不打印原格式，只保留 TCAP

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)
            samples = prepare_sample(samples, sdaa_enabled=sdaa_enabled)
            if samples is None:
                continue
            if not isinstance(samples, dict):
                samples = {"is_empty": True}

            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            # 学习率调度：按原逻辑基于迭代
            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            # 前向
            with torch.sdaa.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                raw_loss = loss_dict.get("loss", loss.detach())  # 未缩放
                scaled_loss = loss / accum_grad_iters

            # 反向
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # 梯度累积更新
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # 更新统计（使用原始 lossDict，不除 accum）
            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # 计算 IPS 与时间
            step_end = time.time()
            step_time = step_end - self._tcap_last_iter_time
            total_time = step_end - self._tcap_start_time

            # 推断 batch 大小（从样本推断常见键）
            batch_size = None
            for key in ["image", "video", "input_ids", "labels"]:
                if key in samples and hasattr(samples[key], "size"):
                    try:
                        batch_size = samples[key].size(0)
                        break
                    except Exception:
                        pass
            if batch_size is None:
                # 回退到配置（避免样本结构不可用时估值为1）
                batch_size = getattr(registry.get("config"), "run_cfg", {}).get("batch_size_train", 1) if registry.has("config") else 1
            effective_bsz = batch_size * accum_grad_iters
            ips = effective_bsz / step_time if step_time > 0 else 0.0

            # 全局迭代号（兼容 iter-based）
            global_iter = global_iter_base + i + 1
            rank = get_rank() if is_dist_avail_and_initialized() else 0
            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            # TCAPPDLL 行
            tcap_line = (
                f"TCAPPDLL {current_time_str} - Epoch: {epoch} "
                f"Iteration: {global_iter} rank : {rank} "
                f"train.loss : {float(raw_loss):.9f} train.ips : {ips:.2f} imgs/s "
                f"train.total_time : {total_time:.6f} step_time : {step_time:.6f}"
            )
            if is_main_process():
                logging.info(tcap_line)
                with open(tcap_log_path, "a") as f:
                    f.write(tcap_line + "\n")

            self._tcap_last_iter_time = step_end

        # 同步与平均
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json
        result_file = os.path.join(result_dir, "%s_rank%d.json" % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, "%s.json" % filename)
        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            result_all = []
            for rank in range(get_world_size()):
                rf = os.path.join(result_dir, "%s_rank%d.json" % (filename, rank))
                res = json.load(open(rf, "r"))
                result_all += res

            if remove_duplicate:
                dedup = []
                ids = set()
                for r in result_all:
                    key = r[remove_duplicate]
                    if key not in ids:
                        ids.add(key)
                        dedup.append(r)
                result_all = dedup

            json.dump(result_all, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
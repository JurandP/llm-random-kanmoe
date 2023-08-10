import os.path
import copy
from typing import Optional, Literal
from types import SimpleNamespace as SN
import math

import numpy as np
import torch
from attr import define

from lizrd.datasets import wikibookdata
import lizrd.datasets.processed_batch
from lizrd.support.logging import AbstractLogger
from research.conditional.moe_layers.continuous_moe import ContinuousMoE
from research.conditional.utils.layer_manager import LayerManager
from research.conditional.utils.misc_tools import get_ith_chunk
from research.conditional.utils.model_utils import make_loss_function
from lizrd.datasets.c4 import NUM_C4_TOKENS


@define(slots=False)
class ConditionalTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_dataloader: wikibookdata.ProcessedDatasetWrapper
    vocab_size: int
    mixed_precision: bool
    logger: Optional[AbstractLogger]
    model_type: Literal["bert", "gpt"]
    logging_interval_loss: int
    logging_interval_light: int
    logging_interval_heavy: int
    total_n_steps: int
    _calculate_loss: Optional[callable] = None
    mask_percent: Optional[float] = None
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    layer_manager: Optional[LayerManager] = None
    loss_accumulator: Optional[float] = None
    n_gpus: int = 1
    hack_name: str = None
    save_weights_path: str = None
    save_weights_interval: int = 1000
    load_weights_path: str = None
    gradient_clipping: float = None
    loss_checkpoint_chungs: int = 0
    gradient_accumulation_steps: int = 1
    lr_decay: float = None
    lr_warmup_steps: int = 0
    lr_decay_interval: int = 0
    lr_decay_type: Literal["step", "cosine"] = "step"
    log_gradients_and_weights: bool = False
    loss_log_intervals: tuple[int] = (1, 10, 100, 1000)

    def __attrs_post_init__(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.loss_accumulators = {
            f"loss_interval/{i}": SN(acc=0.0, interval=i)
            for i in self.loss_log_intervals
        }
        self.loss_accumulators["loss"] = SN(
            acc=0.0, interval=self.logging_interval_loss
        )
        self._calculate_loss = make_loss_function(
            model=self.model_type,
            loss_checkpoint_chungs=self.loss_checkpoint_chungs,
            mask_percentage=self.mask_percent,
        )
        self.layer_manager = LayerManager(
            self.model, self.logging_interval_light, self.logging_interval_heavy
        )
        if self.lr_decay is None:
            self.lr_decay_interval = np.Inf
        if self.lr_decay_type == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda i: i / self.lr_warmup_steps
                if i < self.lr_warmup_steps
                else self.lr_decay ** (i // self.lr_decay_interval),
            )
        elif self.lr_decay_type == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda i: i / self.lr_warmup_steps
                if i < self.lr_warmup_steps
                else 0.5
                * (
                    1
                    + math.cos(
                        math.pi
                        * (i - self.lr_warmup_steps)
                        / (self.total_n_steps - self.lr_warmup_steps)
                    )
                ),
            )

    def _restore_weights(self):
        if self.load_weights_path is not None:
            if os.path.exists(self.load_weights_path):
                print(f"Loading weights from {self.load_weights_path}")
                checkpoint = torch.load(self.load_weights_path)
                self.model.load_state_dict(checkpoint["model"], strict=False)
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scaler.load_state_dict(checkpoint["scaler"])
            else:
                print(
                    f"No weights found at {self.load_weights_path}, training from scratch"
                )

    def _save_weights(self, step):
        if (
            self.save_weights_path is not None
            and step % self.save_weights_interval == 0
        ):
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
            }
            torch.save(checkpoint, self.save_weights_path)
            print(f"Weights saved to {self.save_weights_path} (step {step})")

    def train(self, n_steps: int):
        self._restore_weights()
        for step in range(n_steps + 1):
            if self.hack_name is not None:
                self._hack(self.hack_name, step)
            else:
                self._train_step(step)
            if step % 1000 == 0:
                print(f"Step {step}")

    def _optimize(self, loss, should_apply_gradient=False):
        # since we sum gradients averaged over multiple smaller batches, we need to normalize here
        loss /= self.gradient_accumulation_steps
        if self.gradient_accumulation_steps == 1:
            self.optimizer.zero_grad()
        # clear computation graph, store gradients
        self.scaler.scale(loss).backward()
        if should_apply_gradient:
            if self.gradient_clipping is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )
            self.scaler.step(self.optimizer)
            if self.gradient_accumulation_steps > 1:
                self.optimizer.zero_grad()
            self.scaler.update()

    def _train_step(
        self,
        step,
    ):
        self.model.train()
        if self.logger is not None:
            self.layer_manager.prepare_for_logging(step)
        processed_batch: lizrd.datasets.processed_batch.ProcessedBatch = (
            self.train_dataloader.get_batch()
        )

        loss = self.optimize_with_gradient_accumulation(processed_batch)
        self.lr_scheduler.step()
        if self.logger is not None:
            self._log_train_stats(loss, step)
            self.layer_manager.log(step)
            self._log_heavy(step)
        self._save_weights(step)

    def optimize_with_gradient_accumulation(
        self, processed_batch: wikibookdata.ProcessedBatch
    ):
        """gradient accumulation: slice the batch into minibatches, get gradients from each, then average and apply them"""
        loss_value = 0.0
        for i in range(self.gradient_accumulation_steps):
            batch_copy = copy.deepcopy(processed_batch)
            for tensor in batch_copy:
                tensor.data = get_ith_chunk(
                    tensor.data, self.gradient_accumulation_steps, i
                )
            loss = self._calculate_loss(
                batch=batch_copy,
                model=self.model,
                mixed_precision=self.mixed_precision,
                vocab_size=self.vocab_size,
            )

            # clear computation graph, store gradients, only apply gradients at the end
            should_apply_gradient = i == self.gradient_accumulation_steps - 1
            self._optimize(loss, should_apply_gradient=should_apply_gradient)
            loss_value += loss.item()

        return loss_value

    def _log_train_stats(self, loss_value, step):
        self.logger.report_scalar(title="step", value=step, iteration=step)
        if self.train_dataloader.dataset_type == "c4":
            self._log_fraction_dataset_processed(step)
        self.logger.report_scalar(
            title="lr", value=self.lr_scheduler.get_last_lr()[0], iteration=step
        )
        for name, stats in self.loss_accumulators.items():
            stats.acc += loss_value
            if step % stats.interval == 0 and step > 0:
                self.logger.report_scalar(
                    title=name,
                    value=stats.acc / stats.interval,
                    iteration=step,
                )
                stats.acc = 0.0

    def _log_heavy(self, step):
        g_metrics, w_metrics = {}, {}
        if (
            step % self.logging_interval_heavy == 0
            and step > 0
            and self.log_gradients_and_weights
        ):
            for name, value in self.model.named_parameters():
                if value.grad is not None:
                    norm = torch.linalg.norm(value.grad)
                    g_metrics[f"weight_norms/{name.replace('.', '/')}/grad"] = norm
                if value.requires_grad:
                    norm = torch.linalg.norm(value)
                    w_metrics[f"weight_norms/{name.replace('.', '/')}/weight"] = norm
            g_metrics[f"weight_norms/grad_norm_total"] = torch.linalg.norm(
                torch.tensor(list(g_metrics.values()))
            )
            w_metrics[f"weight_norms/weight_norm_total"] = torch.linalg.norm(
                torch.tensor(list(w_metrics.values()))
            )
            self._log_dict({**g_metrics, **w_metrics}, step)

    def _log_dict(self, metrics, step):
        for k, v in metrics.items():
            self.logger.report_scalar(title=k, value=v, iteration=step)

    def _log_fraction_dataset_processed(self, step):
        batch_size = self.train_dataloader.batch_size
        seq_len = self.train_dataloader.sequence_length
        processed = step * batch_size * seq_len
        total = NUM_C4_TOKENS
        self.logger.report_scalar(
            title="Fraction of dataset that is processed (assumuing no DDP)",
            value=processed / total,
            iteration=step,
        )

    def _hack(self, hack_name, step):
        if hack_name == "batch_size":
            self._hack_for_batch_size(step)
        elif hack_name == "expert_size":
            self._hack_for_contmoe_expert_size(step)
        else:
            raise ValueError(f"Unknown hack {hack_name}")

    def _hack_for_batch_size(
        self,
        step,
    ):
        """
        This is a hack to easily determine the maximal batch size that can be used with given GPU memory and model size.
        """
        self.model.train()
        processed_batch = self.train_dataloader.get_batch()
        assert isinstance(
            processed_batch, lizrd.datasets.processed_batch.ProcessedBatch
        )
        for tensor in processed_batch:
            tensor.data = tensor[:1].repeat(step + 1, 1).data
        loss = self._calculate_loss(
            batch=processed_batch,
            model=self.model,
            mixed_precision=self.mixed_precision,
            vocab_size=self.vocab_size,
        )
        self._optimize(loss, should_apply_gradient=True)
        if self.logger is not None:
            self.logger.report_scalar(
                title="max batch size", value=step * self.n_gpus, iteration=step
            )

    def _hack_for_contmoe_expert_size(
        self,
        step,
    ):
        """
        This is a hack to easily determine the maximal batch size that can be used with given GPU memory and model size.
        """
        assert all(
            [
                isinstance(layer, ContinuousMoE)
                for name, layer in self.layer_manager._layers
            ]
        )
        self.model.train()
        processed_batch = self.train_dataloader.get_batch()
        assert isinstance(
            processed_batch, lizrd.datasets.processed_batch.ProcessedBatch
        )
        for block_name, layer in self.layer_manager._layers:
            layer.expertsize = step + 1
            layer.init_parameters()
            layer.to(torch.device("cuda"))
        loss = self._calculate_loss(
            batch=processed_batch,
            model=self.model,
            mixed_precision=self.mixed_precision,
            vocab_size=self.vocab_size,
        )
        self._optimize(loss, should_apply_gradient=True)
        self.logger.report_scalar(title="max expert size", value=step, iteration=step)

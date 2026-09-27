from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainConfig
from train.visualizer import TrainingVisualizer


class PretrainingTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
    ) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dirs = {
            category: self.output_dir / "checkpoints" / category
            for category in ("periodic", "best", "threshold", "chunk", "final")
        }
        for directory in self.checkpoint_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        self.device = self._resolve_device(config.device)
        self.model = model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100)
        self.visualizer = TrainingVisualizer(config)

        self.global_step = 0
        self.tokens_seen = 0
        self.samples_seen = 0

        self.best_checkpoint_value: float | None = None
        self.early_stop_best_value: float | None = None
        self.early_stop_best_step = 0
        self.threshold_checkpoint_saved = False

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device(requested)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    @staticmethod
    def _is_improvement(
        value: float,
        best_value: float | None,
        mode: str,
        min_delta: float,
    ) -> bool:
        if best_value is None:
            return True
        if mode == "min":
            return value < best_value - min_delta
        return value > best_value + min_delta

    @staticmethod
    def _reaches_threshold(value: float, threshold: float, mode: str) -> bool:
        if mode == "min":
            return value <= threshold
        return value >= threshold

    def _parameter_norm(self) -> float:
        squared_sum = 0.0
        with torch.no_grad():
            for parameter in self.model.parameters():
                squared_sum += float(parameter.detach().float().pow(2).sum().item())
        return squared_sum**0.5

    def save_checkpoint(
        self,
        category: str,
        name: str,
        chunk_id: int,
        monitored_metric: str | None = None,
        monitored_value: float | None = None,
    ) -> Path:
        if category not in self.checkpoint_dirs:
            raise ValueError(f"Unknown checkpoint category: {category}")

        checkpoint_path = self.checkpoint_dirs[category] / name
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

        payload: dict[str, object] = {
            "model_state_dict": self.model.state_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "samples_seen": self.samples_seen,
            "chunk_id": chunk_id,
            "best_checkpoint_value": self.best_checkpoint_value,
            "early_stop_best_value": self.early_stop_best_value,
            "early_stop_best_step": self.early_stop_best_step,
            "threshold_checkpoint_saved": self.threshold_checkpoint_saved,
            "monitored_metric": monitored_metric,
            "monitored_value": monitored_value,
            "config": self.config.to_dict(),
        }
        if self.config.save_optimizer_state:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()

        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if self.config.save_optimizer_state and optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        self.global_step = int(checkpoint.get("global_step", 0))
        self.tokens_seen = int(checkpoint.get("tokens_seen", 0))
        self.samples_seen = int(checkpoint.get("samples_seen", 0))
        self.best_checkpoint_value = checkpoint.get("best_checkpoint_value")
        self.early_stop_best_value = checkpoint.get("early_stop_best_value")
        self.early_stop_best_step = int(
            checkpoint.get("early_stop_best_step", self.global_step)
        )
        self.threshold_checkpoint_saved = bool(
            checkpoint.get("threshold_checkpoint_saved", False)
        )

    def _check_fixed_step_stop(self) -> str | None:
        if self.config.early_stop_step > 0 and self.global_step >= self.config.early_stop_step:
            return "early_stop_step"
        if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
            return "max_steps"
        return None

    def _handle_metric_actions(
        self,
        metrics: dict[str, float | int],
        smoothed_metrics: dict[str, float],
        chunk_id: int,
    ) -> str | None:
        def monitored_value(metric_name: str) -> float | None:
            if metric_name in smoothed_metrics:
                return float(smoothed_metrics[metric_name])
            value = metrics.get(metric_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            return None

        if self.config.save_best_checkpoint:
            value = monitored_value(self.config.best_checkpoint_metric)
            if value is not None and self._is_improvement(
                value,
                self.best_checkpoint_value,
                self.config.best_checkpoint_mode,
                self.config.best_checkpoint_min_delta,
            ):
                self.best_checkpoint_value = value
                safe_metric = self.config.best_checkpoint_metric.replace("/", "_")
                self.save_checkpoint(
                    category="best",
                    name=f"best-{safe_metric}.pt",
                    chunk_id=chunk_id,
                    monitored_metric=self.config.best_checkpoint_metric,
                    monitored_value=value,
                )

        if (
            self.config.save_threshold_checkpoint
            and self.config.threshold_checkpoint_value is not None
            and (
                not self.config.threshold_checkpoint_once
                or not self.threshold_checkpoint_saved
            )
        ):
            value = monitored_value(self.config.threshold_checkpoint_metric)
            if value is not None and self._reaches_threshold(
                value,
                self.config.threshold_checkpoint_value,
                self.config.threshold_checkpoint_mode,
            ):
                safe_metric = self.config.threshold_checkpoint_metric.replace("/", "_")
                self.threshold_checkpoint_saved = True
                self.save_checkpoint(
                    category="threshold",
                    name=f"{safe_metric}-step-{self.global_step:08d}.pt",
                    chunk_id=chunk_id,
                    monitored_metric=self.config.threshold_checkpoint_metric,
                    monitored_value=value,
                )

        if self.global_step < self.config.early_stop_warmup_steps:
            return None

        early_value = monitored_value(self.config.early_stop_metric)
        if early_value is None:
            return None

        if self._is_improvement(
            early_value,
            self.early_stop_best_value,
            self.config.early_stop_mode,
            self.config.early_stop_min_delta,
        ):
            self.early_stop_best_value = early_value
            self.early_stop_best_step = self.global_step

        if (
            self.config.early_stop_threshold is not None
            and self._reaches_threshold(
                early_value,
                self.config.early_stop_threshold,
                self.config.early_stop_mode,
            )
        ):
            return "metric_threshold"

        if (
            self.config.early_stop_patience_steps > 0
            and self.global_step - self.early_stop_best_step
            >= self.config.early_stop_patience_steps
        ):
            return "metric_patience"

        return None

    def train_chunk(
        self,
        train_loader: DataLoader,
        chunk_id: int,
    ) -> dict[str, float | int | str | bool]:
        fixed_reason = self._check_fixed_step_stop()
        if fixed_reason is not None:
            return {
                "chunk_id": chunk_id,
                "chunk_steps": 0,
                "global_step": self.global_step,
                "last_loss": float("nan"),
                "completed_chunk": False,
                "should_stop": True,
                "stop_reason": fixed_reason,
            }

        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        expected_steps = len(train_loader) * self.config.epochs
        chunk_steps = 0
        last_loss = float("nan")
        stop_reason: str | None = None

        interval_loss_sum = 0.0
        interval_grad_norm_sum = 0.0
        interval_steps = 0
        interval_tokens = 0
        interval_samples = 0
        interval_started_at = time.perf_counter()

        def emit_metrics(epoch_index: int) -> str | None:
            nonlocal interval_loss_sum
            nonlocal interval_grad_norm_sum
            nonlocal interval_steps
            nonlocal interval_tokens
            nonlocal interval_samples
            nonlocal interval_started_at

            if interval_steps == 0:
                return None

            elapsed = max(time.perf_counter() - interval_started_at, 1e-9)
            train_loss = interval_loss_sum / interval_steps
            grad_norm = interval_grad_norm_sum / interval_steps
            learning_rate = float(self.optimizer.param_groups[0]["lr"])

            metrics: dict[str, float | int] = {
                "global_step": self.global_step,
                "chunk_id": chunk_id,
                "epoch": epoch_index + 1,
                "train_loss": train_loss,
                "perplexity": math.exp(min(train_loss, 20.0)),
                "grad_norm": grad_norm,
                "learning_rate": learning_rate,
                "tokens_per_second": interval_tokens / elapsed,
                "samples_per_second": interval_samples / elapsed,
                "tokens_seen": self.tokens_seen,
                "samples_seen": self.samples_seen,
            }

            if self.config.log_parameter_norm:
                metrics["parameter_norm"] = self._parameter_norm()

            if self.device.type == "cuda":
                metrics.update(
                    {
                        "gpu_allocated_gb": torch.cuda.memory_allocated(self.device)
                        / 1024**3,
                        "gpu_reserved_gb": torch.cuda.memory_reserved(self.device)
                        / 1024**3,
                        "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated(
                            self.device
                        )
                        / 1024**3,
                    }
                )

            smoothed = self.visualizer.record(metrics, model=self.model)
            reason = self._handle_metric_actions(
                metrics=metrics,
                smoothed_metrics=smoothed,
                chunk_id=chunk_id,
            )

            interval_loss_sum = 0.0
            interval_grad_norm_sum = 0.0
            interval_steps = 0
            interval_tokens = 0
            interval_samples = 0
            interval_started_at = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            return reason

        for epoch in range(self.config.epochs):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=True
                )

                self.optimizer.zero_grad(set_to_none=True)

                logits, aux_loss, router_infos = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                )

                lm_loss = self.loss_function(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

                loss = (
                    lm_loss
                    + self.config.moe_aux_loss_coef * aux_loss
                )

                loss.backward()
                
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.max_grad_norm,
                )
                self.optimizer.step()

                self.global_step += 1
                chunk_steps += 1
                last_loss = float(loss.item())

                batch_tokens = int(labels.ne(-100).sum().item())
                batch_samples = int(input_ids.size(0))
                self.tokens_seen += batch_tokens
                self.samples_seen += batch_samples

                interval_loss_sum += last_loss
                interval_grad_norm_sum += float(gradient_norm)
                interval_steps += 1
                interval_tokens += batch_tokens
                interval_samples += batch_samples

                if self.global_step % self.config.log_every_steps == 0:
                    print(
                        f"chunk={chunk_id} "
                        f"epoch={epoch + 1}/{self.config.epochs} "
                        f"step={self.global_step} "
                        f"loss={last_loss:.4f} "
                        f"grad_norm={float(gradient_norm):.4f}"
                    )

                if (
                    self.config.save_every_steps > 0
                    and self.global_step % self.config.save_every_steps == 0
                ):
                    self.save_checkpoint(
                        category="periodic",
                        name=f"checkpoint-step-{self.global_step:08d}.pt",
                        chunk_id=chunk_id,
                    )

                if self.global_step % self.config.metrics_every_steps == 0:
                    stop_reason = emit_metrics(epoch)

                if stop_reason is None:
                    stop_reason = self._check_fixed_step_stop()

                if stop_reason is not None:
                    break

            if stop_reason is not None:
                break

        if interval_steps > 0:
            metric_stop_reason = emit_metrics(min(self.config.epochs - 1, epoch))
            if stop_reason is None:
                stop_reason = metric_stop_reason

        completed_chunk = chunk_steps >= expected_steps
        return {
            "chunk_id": chunk_id,
            "chunk_steps": chunk_steps,
            "global_step": self.global_step,
            "last_loss": last_loss,
            "completed_chunk": completed_chunk,
            "should_stop": stop_reason is not None,
            "stop_reason": stop_reason or "",
        }

    def finalize(
        self,
        chunk_id: int,
        stop_reason: str,
        trained_documents: int,
    ) -> dict[str, float | int | str]:
        final_checkpoint = self.save_checkpoint(
            category="final",
            name="final_checkpoint.pt",
            chunk_id=chunk_id,
        )
        self.visualizer.close()

        summary = {
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "samples_seen": self.samples_seen,
            "trained_documents": trained_documents,
            "device": str(self.device),
            "stop_reason": stop_reason,
            "final_checkpoint": str(final_checkpoint),
        }
        (self.output_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

"""Optional Weights & Biases reporting for competition training."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import TrainingConfig


_PERFORMANCE_METRICS = {
    "wall_seconds",
    "samples_per_second",
    "rollout_seconds",
    "update_seconds",
}


def _training_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Give W&B metrics stable namespaces without changing the JSONL schema."""
    metrics: dict[str, Any] = {"iteration": record["iteration"]}
    for name, value in record.items():
        if name == "iteration":
            continue
        if name == "stage":
            metrics["curriculum/stage"] = value
        elif name in _PERFORMANCE_METRICS:
            metrics[f"performance/{name}"] = value
        else:
            metrics[f"training/{name}"] = value
    return metrics


class WandbTracker:
    """A failure-tolerant, host-only W&B metrics sink.

    JSONL remains the durable record. If W&B cannot initialize or stops
    accepting metrics, training continues and the local log remains complete.
    """

    def __init__(self, run=None):
        self._run = run

    @classmethod
    def start(
        cls,
        config: TrainingConfig,
        *,
        start_iteration: int,
        resume: str | None,
        device_count: int,
    ) -> WandbTracker:
        project = config.wandb_project or os.environ.get("WANDB_PROJECT")
        if not project or os.environ.get("WANDB_MODE", "").lower() == "disabled":
            return cls()

        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "W&B tracking is enabled but the 'wandb' package is not installed. "
                "Install it with: pip install -e '.[train,tracking]'"
            ) from error

        run_config = asdict(config)
        run_config.update(
            {
                "start_iteration": start_iteration,
                "resume_checkpoint": Path(resume).name if resume else None,
                "device_count": device_count,
            }
        )
        run_name = config.run_name
        if resume:
            run_name = f"{run_name}-resume-{start_iteration:06d}"

        try:
            run = wandb.init(
                project=project,
                entity=config.wandb_entity,
                name=run_name,
                group=config.wandb_group or config.run_name,
                tags=list(config.wandb_tags) or None,
                job_type="training",
                config=run_config,
            )
            run.define_metric("iteration")
            run.define_metric("*", step_metric="iteration")
            run.define_metric("evaluation/score", summary="max")
            print(f"W&B run: {run.url}")
            return cls(run)
        except Exception as error:  # noqa: BLE001 - W&B is a non-critical sink.
            print(
                f"Warning: W&B initialization failed; continuing locally: {error}",
                file=sys.stderr,
            )
            return cls()

    @property
    def active(self) -> bool:
        return self._run is not None

    def _log(self, metrics: dict[str, Any]) -> None:
        if self._run is None:
            return
        try:
            self._run.log(metrics)
        except Exception as error:  # noqa: BLE001 - preserve JSONL on service failure.
            print(
                f"Warning: W&B logging failed; disabling it for this run: {error}",
                file=sys.stderr,
            )
            self._run = None

    def log_training(self, record: dict[str, Any]) -> None:
        self._log(_training_metrics(record))

    def log_evaluation(self, record: dict[str, Any]) -> None:
        self._log(record)

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception as error:  # noqa: BLE001 - shutdown cannot fail training.
            print(f"Warning: W&B shutdown failed: {error}", file=sys.stderr)
        finally:
            self._run = None

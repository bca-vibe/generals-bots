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
    "active_training_seconds",
    "iteration_seconds",
    "samples_per_second",
    "samples_per_gpu_second",
    "allocated_gpu_hours",
    "training_gpu_hours",
    "rollout_seconds",
    "update_seconds",
    "host_seconds",
}

_PPO_METRICS = {
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approximate_kl",
    "clip_fraction",
    "gradient_norm",
    "epochs_used",
}

_ROLLOUT_METRICS = {"episodes", "wins", "losses", "draws", "score", "mean_reward"}


def _training_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Give W&B metrics stable namespaces without changing the JSONL schema."""
    metrics: dict[str, Any] = {"iteration": record["iteration"]}
    for name, value in record.items():
        if name == "iteration":
            continue
        if name == "stage":
            metrics["curriculum/stage"] = value
        elif name == "cumulative_samples":
            metrics["progress/cumulative_samples"] = value
        elif name in {"continuation_iteration", "continuation_samples"}:
            metrics[f"progress/{name}"] = value
        elif name in _PERFORMANCE_METRICS:
            metrics[f"performance/{name}"] = value
        elif name in _PPO_METRICS:
            canonical = "approx_kl" if name == "approximate_kl" else name
            metrics[f"ppo/{canonical}"] = value
        elif name in _ROLLOUT_METRICS:
            metrics[f"rollout/{name}"] = value
        elif name == "raw_advantage_std":
            metrics["advantages/raw_std"] = value
        elif name == "explained_variance":
            metrics["value/explained_variance"] = value
        elif name in {"learning_rate", "entropy_coefficient"}:
            metrics[f"optimization/{name}"] = value
        else:
            metrics[f"training/{name}"] = value
    return metrics


class WandbTracker:
    """A failure-tolerant, host-only W&B metrics sink.

    JSONL remains the durable record. If W&B cannot initialize or stops
    accepting metrics, training continues and the local log remains complete.
    """

    def __init__(self, run=None, wandb_module=None):
        self._run = run
        self._wandb = wandb_module
        self._checkpoint_rows: dict[int, dict[str, Any]] = {}

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
        run_name = config.wandb_run_name or config.run_name
        if resume and not config.wandb_run_id:
            run_name = f"{run_name}-resume-{start_iteration:06d}"

        try:
            run = wandb.init(
                project=project,
                entity=config.wandb_entity,
                name=run_name,
                group=config.wandb_group or config.run_name,
                tags=list(config.wandb_tags) or None,
                job_type=config.wandb_job_type,
                config=run_config,
                id=config.wandb_run_id,
                resume="allow" if config.wandb_run_id else None,
            )
            run.define_metric("iteration")
            run.define_metric("*", step_metric="iteration")
            run.define_metric("evaluation/score", summary="max")
            print(f"W&B run: {run.url}")
            return cls(run, wandb)
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

    def log_initialization(self, record: dict[str, Any]) -> None:
        self._log(record)

    def log_checkpoint_export(self, record: dict[str, Any]) -> None:
        """Mirror an asynchronous checkpoint publication event into W&B."""
        metrics = {
            "iteration": record["iteration"],
            "checkpoint/hf_export_requested": int(record.get("requested", False)),
            "checkpoint/hf_export_complete": int(record.get("complete", False)),
            "checkpoint/hf_hash_verified": int(record.get("hash_verified", False)),
            "checkpoint/competition_bundle_available": int(
                record.get("competition_bundle_available", False)
            ),
        }
        if "upload_seconds" in record:
            metrics["checkpoint/hf_upload_seconds"] = record["upload_seconds"]
        for source, destination in (
            ("checkpoint_bytes", "checkpoint/full_bytes"),
            ("competition_bytes", "checkpoint/competition_bytes"),
            ("checkpoint_sha256", "checkpoint/full_sha256"),
            ("competition_sha256", "checkpoint/competition_sha256"),
            ("remote_checkpoint_path", "checkpoint/hf_checkpoint_path"),
            ("remote_competition_path", "checkpoint/hf_competition_path"),
        ):
            if source in record:
                metrics[destination] = record[source]
        self._log(metrics)
        if self._run is None or self._wandb is None:
            return
        iteration = int(record["iteration"])
        row = self._checkpoint_rows.setdefault(iteration, {"iteration": iteration})
        row.update({name: value for name, value in record.items() if value is not None})
        columns = [
            "iteration",
            "checkpoint_sha256",
            "competition_sha256",
            "remote_checkpoint_path",
            "remote_competition_path",
            "hash_verified",
        ]
        data = [
            [row.get(column, "") for column in columns]
            for _, row in sorted(self._checkpoint_rows.items())
        ]
        try:
            table = self._wandb.Table(
                columns=columns,
                data=data,
            )
            self._run.log(
                {"iteration": record["iteration"], "checkpoint/exports": table}
            )
        except Exception as error:  # noqa: BLE001 - tabular display is optional.
            print(f"Warning: W&B checkpoint table update failed: {error}", file=sys.stderr)

    def update_summary(self, values: dict[str, Any]) -> None:
        if self._run is None:
            return
        try:
            self._run.summary.update(values)
        except Exception as error:  # noqa: BLE001
            print(f"Warning: W&B summary update failed: {error}", file=sys.stderr)

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception as error:  # noqa: BLE001 - shutdown cannot fail training.
            print(f"Warning: W&B shutdown failed: {error}", file=sys.stderr)
        finally:
            self._run = None

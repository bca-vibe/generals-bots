"""Verify authenticated W&B lineage and metric writes before the castle A/B."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import TrainingConfig
from .tracking import WandbTracker


def run(config_paths: list[Path], output: Path) -> dict:
    records = []
    for path in config_paths:
        production = TrainingConfig.from_toml(path)
        smoke_id = f"{production.wandb_run_id}-preflight"
        smoke = replace(
            production,
            run_name=f"preflight_{production.run_name}",
            wandb_run_id=smoke_id,
            wandb_run_name=f"PRELAUNCH CHECK · {production.wandb_run_name}",
            wandb_job_type="preflight-validation",
            wandb_tags=(*production.wandb_tags, "preflight-validation"),
        )
        tracker = WandbTracker.start(
            smoke,
            start_iteration=production.parent_final_iteration,
            resume=production.resume_checkpoint_source,
            device_count=4,
        )
        if not tracker.active:
            raise RuntimeError(f"W&B did not initialize for {path}")
        url = tracker._run.url  # noqa: SLF001 - preflight must record the URL.
        tracker.log_initialization(
            {
                "iteration": production.parent_final_iteration,
                "preflight/wandb_authenticated": 1,
                "preflight/parent_iteration": production.parent_final_iteration,
                "preflight/parent_samples": production.parent_final_samples,
                "preflight/arm": production.run_name,
            }
        )
        tracker.update_summary(
            {
                "preflight_status": "passed",
                "production_run_id": production.wandb_run_id,
                "production_group": production.wandb_group,
            }
        )
        tracker.finish()
        records.append(
            {
                "config": str(path),
                "preflight_run_id": smoke_id,
                "preflight_url": url,
                "production_run_id": production.wandb_run_id,
                "group": production.wandb_group,
                "parent_run_id": production.parent_wandb_run_id,
                "parent_iteration": production.parent_final_iteration,
                "parent_samples": production.parent_final_samples,
            }
        )
    result = {"status": "passed", "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.configs, args.output)

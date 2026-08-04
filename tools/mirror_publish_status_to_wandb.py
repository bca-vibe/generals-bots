"""Replay durable checkpoint publication state into a continuation W&B run."""

from __future__ import annotations

import argparse
import json

from generals.training.config import TrainingConfig
from generals.training.tracking import WandbTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_toml(args.config)
    tracker = WandbTracker.start(
        config,
        start_iteration=config.parent_final_iteration,
        resume=config.league_checkpoint_path,
        device_count=8,
    )
    for request_path in sorted((config.run_dir / "publish_requests").glob("*.json")):
        tracker.log_checkpoint_export(
            json.loads(request_path.read_text(encoding="utf-8"))
        )
    status_path = config.run_dir / "publish_status.jsonl"
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            tracker.log_checkpoint_export(json.loads(line))
    tracker.finish()


if __name__ == "__main__":
    main()

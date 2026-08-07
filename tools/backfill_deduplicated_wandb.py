#!/usr/bin/env python3
"""Create a clean W&B training run from a last-record-wins metrics JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import wandb
from wandb.errors import CommError

from generals.training.tracking import _training_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--source-run-url", required=True)
    return parser.parse_args()


def deduplicate(path: Path) -> tuple[list[dict[str, Any]], int]:
    by_iteration: dict[int, dict[str, Any]] = {}
    total = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        record = json.loads(line)
        if "iteration" not in record:
            raise ValueError(f"Missing iteration at {path}:{line_number}")
        by_iteration[int(record["iteration"])] = record
    if not by_iteration:
        raise ValueError(f"No metric records found in {path}")
    return [by_iteration[key] for key in sorted(by_iteration)], total


def assert_destination_absent(entity: str, project: str, run_id: str) -> None:
    try:
        existing = wandb.Api(timeout=30).run(f"{entity}/{project}/{run_id}")
    except CommError:
        return
    raise RuntimeError(
        f"Refusing to append to existing destination run: {existing.url}. "
        "Choose a new --run-id."
    )


def main() -> None:
    args = parse_args()
    records, source_rows = deduplicate(args.metrics)
    assert_destination_absent(args.entity, args.project, args.run_id)

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        name=args.run_name,
        group=args.group,
        job_type="posthoc-deduplicated-training-history",
        tags=["deduplicated", "host-loss-recovery", "posthoc"],
        config={
            "source_run_url": args.source_run_url,
            "source_metrics": str(args.metrics),
            "deduplication": "last_record_wins_by_iteration",
            "source_rows": source_rows,
            "deduplicated_rows": len(records),
            "first_iteration": int(records[0]["iteration"]),
            "last_iteration": int(records[-1]["iteration"]),
        },
        resume="never",
    )
    run.define_metric("iteration")
    run.define_metric("*", step_metric="iteration")
    for record in records:
        run.log(_training_metrics(record))
    run.summary.update(
        {
            "deduplication/source_rows": source_rows,
            "deduplication/retained_rows": len(records),
            "deduplication/removed_rows": source_rows - len(records),
            "deduplication/source_run_url": args.source_run_url,
        }
    )
    url = run.url
    run.finish()
    print(
        json.dumps(
            {
                "destination_url": url,
                "source_rows": source_rows,
                "retained_rows": len(records),
                "removed_rows": source_rows - len(records),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

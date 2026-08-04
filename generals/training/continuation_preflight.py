"""One-iteration 8-device resume preflight including checkpoint and league paths."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import jax

from .config import TrainingConfig
from .train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    production = TrainingConfig.from_toml(args.config)
    if jax.device_count() != 8:
        raise RuntimeError(f"Preflight requires exactly 8 devices, got {jax.device_count()}")
    preflight = replace(
        production,
        run_name=production.run_name + "_preflight",
        num_iterations=production.parent_final_iteration + 1,
        checkpoint_every=1,
        latest_checkpoint_every=1,
        league_eval_every=1,
        league_eval_maps=8,
        league_checkpoint_maps=8,
        league_opponents=("random",),
        league_eval_after_training=False,
        reset_pool_every=0,
        wandb_project=None,
        wandb_run_id=None,
        wandb_run_name=None,
    )
    preflight.validate()
    train(preflight, resume=args.resume)
    metadata_path = preflight.run_dir / "terminal_checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["iteration"] != production.parent_final_iteration + 1:
        raise RuntimeError(f"Preflight stopped at unexpected iteration: {metadata}")
    league_path = preflight.run_dir / f"league_{metadata['iteration']:06d}.json"
    if not league_path.is_file():
        raise RuntimeError("Preflight did not produce its periodic league result")
    request_path = (
        preflight.run_dir
        / "publish_requests"
        / f"checkpoint_{metadata['iteration']:06d}.json"
    )
    if not request_path.is_file():
        raise RuntimeError("Preflight did not produce its publication request")
    summary = {
        "checkpoint": metadata,
        "devices": [str(device) for device in jax.devices()],
        "league": str(league_path),
        "publication_request": str(request_path),
    }
    (preflight.run_dir / "preflight_success.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

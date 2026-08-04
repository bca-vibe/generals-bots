"""Evaluate historical checkpoints against the locked competition league."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .config import TrainingConfig
from .tracking import WandbTracker
from .train import _learning_rate, _load_checkpoint_state, _run_league, build_network

DEFAULT_OPPONENTS = (
    "boss",
    "random",
    "hunter",
    "harvester",
    "raider",
    "deathtouch_clock",
)


def load_checkpoint(config: TrainingConfig, path: Path):
    key = jax.random.PRNGKey(config.seed)
    key, network_key = jax.random.split(key)
    network = build_network(config, network_key)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(partial(_learning_rate, config)),
    )
    optimizer_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    skeleton = (
        network,
        optimizer_state,
        network,
        jnp.int32(0),
        jnp.int32(0),
        key,
    )
    return _load_checkpoint_state(path, skeleton, config)


def evaluate_checkpoints(
    config_path: Path,
    checkpoints: list[Path],
    output_dir: Path,
) -> None:
    base = TrainingConfig.from_toml(config_path)
    config = replace(
        base,
        run_name="arch_ab_d448_8xh100_5h_retry1_20260803_original_baselines",
        output_dir=str(output_dir.parent),
        wandb_project="generals-bots",
        wandb_group="arch_ab_d448_8xh100_5h_retry1_20260803",
        wandb_run_id="arch-ab-d448-5h-retry1-original-baselines-20260803",
        wandb_tags=("architecture-ab", "retry1", "original-baselines", "legacy-38"),
        league_eval_maps=256,
        league_eval_seed=1044,
        league_opponents=DEFAULT_OPPONENTS,
        league_eval_policies=("raw", "ema"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = WandbTracker.start(
        config,
        start_iteration=0,
        resume=None,
        device_count=jax.device_count(),
    )
    index: list[dict] = []
    for checkpoint in checkpoints:
        raw, _, ema, iteration, stage_index, _ = load_checkpoint(config, checkpoint)
        iteration_int = int(iteration)
        payload = _run_league(
            config,
            {"raw": raw, "ema": ema},
            tracker,
            output_dir,
            iteration_int,
            label=f"original_{iteration_int:06d}",
        )
        index.append(
            {
                "checkpoint": str(checkpoint),
                "iteration": iteration_int,
                "historical_curriculum_stage": int(stage_index),
                "observation_schema": config.observation_schema,
                "map_generator_provenance": "original_pre_fix",
                "results": payload,
            }
        )
    (output_dir / "original_baseline_league.json").write_text(
        json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
    )
    tracker.update_summary({"evaluated_checkpoints": len(index)})
    tracker.finish()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_checkpoints(args.config, args.checkpoints, args.output_dir)


if __name__ == "__main__":
    main()

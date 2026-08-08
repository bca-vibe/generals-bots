"""Evaluate 20k EMA against 20k raw and 19k EMA on the league map set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from round_robin import combine_shards, scalar_result, sha256


EXPECTED_SHA256 = {
    19000: "5cb1ee6ec929f9c144751f51639baa850a4f7fb79febd8351157daac9ad0f2ae",
    20000: "2e93206823032dc2a5ab5d21e1af5a580c68e131e4c4a1674950c4ec080b3357",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps", type=int, default=512)
    parser.add_argument("--shard-maps", type=int, default=128)
    parser.add_argument("--map-seed", type=int, default=202608076)
    args = parser.parse_args()
    if args.maps % args.shard_maps:
        raise ValueError("maps must be divisible by shard-maps")

    import sys

    sys.path.insert(0, str(args.repo))
    from generals.training.config import TrainingConfig
    from generals.training.evaluate_head_to_head import _load_policy
    from generals.training.evaluation import evaluate_paired_networks
    from generals.training.train import make_environment

    config = TrainingConfig.from_toml(args.config)
    checkpoints = {
        iteration: args.checkpoint_root / f"checkpoint_{iteration:06d}.eqx"
        for iteration in EXPECTED_SHA256
    }
    for iteration, checkpoint in checkpoints.items():
        digest = sha256(checkpoint)
        if digest != EXPECTED_SHA256[iteration]:
            raise RuntimeError(
                f"checkpoint hash mismatch for {iteration}: {digest}"
            )

    networks = {}
    metadata = []
    for iteration, policy in ((20000, "ema"), (20000, "raw"), (19000, "ema")):
        network, loaded_iteration, stage = _load_policy(
            config, checkpoints[iteration], policy
        )
        if loaded_iteration != iteration:
            raise RuntimeError(
                f"checkpoint {iteration} contains iteration {loaded_iteration}"
            )
        name = f"c{iteration}_{policy}"
        networks[name] = network
        metadata.append(
            {
                "name": name,
                "iteration": iteration,
                "policy": policy,
                "curriculum_stage": stage,
                "checkpoint": str(checkpoints[iteration]),
                "checkpoint_sha256": EXPECTED_SHA256[iteration],
                "inference": "greedy",
            }
        )

    environment = make_environment(
        config, config.curriculum[-1], pool_size=max(args.maps, 16)
    )
    pool, _ = environment.reset(jax.random.PRNGKey(args.map_seed))
    pool = jax.tree.map(lambda value: value[: args.maps], pool)
    pool = pool._replace(pool_idx=jnp.arange(args.maps, dtype=jnp.int32))

    @eqx.filter_jit
    def evaluate(pool_shard, network_a, network_b):
        return evaluate_paired_networks(
            environment,
            pool_shard,
            network_a,
            network_b,
            args.shard_maps,
            config.truncation,
            schema_a=config.observation_schema,
            schema_b=config.observation_schema,
            pad_to=config.pad_to,
            history_size=config.history_size,
            temporal_window=config.temporal_window,
            sampling="greedy",
        )

    matches = []
    for first, second in (
        ("c20000_ema", "c20000_raw"),
        ("c20000_ema", "c19000_ema"),
    ):
        started = time.perf_counter()
        shards = []
        for start in range(0, args.maps, args.shard_maps):
            pool_shard = jax.tree.map(
                lambda value: value[start : start + args.shard_maps], pool
            )
            pool_shard = pool_shard._replace(
                pool_idx=jnp.arange(args.shard_maps, dtype=jnp.int32)
            )
            shards.append(
                scalar_result(
                    evaluate(pool_shard, networks[first], networks[second])
                )
            )
        result = combine_shards(shards, args.maps)
        result.update(
            {
                "a": first,
                "b": second,
                "evaluation_seconds": time.perf_counter() - started,
            }
        )
        matches.append(result)
        print(
            f"{first} vs {second}: {int(result['wins'])}W/"
            f"{int(result['losses'])}L/{int(result['draws'])}D "
            f"score={result['score']:.4f} "
            f"CI=[{result['score_ci95'][0]:.4f},{result['score_ci95'][1]:.4f}]",
            flush=True,
        )

    payload = {
        "schema": "checkpoint_20k_followup_v1",
        "map_seed": args.map_seed,
        "maps_per_matchup": args.maps,
        "games_per_matchup": 2 * args.maps,
        "shard_maps": args.shard_maps,
        "locked_maps_across_matchups": True,
        "seat_swapped": True,
        "inference": "greedy",
        "participants": metadata,
        "matches": matches,
        "total_games": int(sum(match["games"] for match in matches)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

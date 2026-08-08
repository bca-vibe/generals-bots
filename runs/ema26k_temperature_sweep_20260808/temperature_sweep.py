"""Paired-map temperature round robin for one exported checkpoint policy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from round_robin import (
    Participant,
    StrategyNetwork,
    combine_shards,
    load_bot,
    load_parameters,
    scalar_result,
    sha256,
    write_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--agent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--temperatures", type=float, nargs="+", required=True)
    parser.add_argument("--maps", type=int, default=512)
    parser.add_argument("--shard-maps", type=int, default=128)
    parser.add_argument("--map-seed", type=int, required=True)
    parser.add_argument("--action-seed", type=int, required=True)
    args = parser.parse_args()
    if args.maps % args.shard_maps:
        raise ValueError("maps must be divisible by shard-maps")
    if len(set(args.temperatures)) != len(args.temperatures):
        raise ValueError("temperatures must be unique")
    if any(temperature < 0 for temperature in args.temperatures):
        raise ValueError("temperatures must be nonnegative; zero encodes greedy")

    import sys

    sys.path.insert(0, str(args.repo))
    from generals.training.config import TrainingConfig
    from generals.training.evaluation import evaluate_paired_networks
    from generals.training.train import make_environment

    directory = args.agent_root / f"c{args.checkpoint}"
    bot = load_bot(directory / "bot.py")
    metadata = json.loads((directory / "export_metadata.json").read_text())
    weights_path = directory / "weights.npz"
    weights_digest = sha256(weights_path)
    if weights_digest != metadata["weights_sha256"]:
        raise RuntimeError("weights hash mismatch")
    if metadata["iteration"] != args.checkpoint:
        raise RuntimeError(
            f"export iteration {metadata['iteration']} != {args.checkpoint}"
        )
    parameters = load_parameters(weights_path)
    participants = []
    for temperature in sorted(args.temperatures):
        greedy = temperature == 0.0
        suffix = "greedy" if greedy else f"t{temperature:g}"
        participants.append(
            Participant(
                name=f"c{args.checkpoint}_{metadata['policy']}_{suffix}",
                checkpoint=args.checkpoint,
                temperature=None if greedy else temperature,
                network=StrategyNetwork(
                    parameters,
                    bot._policy_logits,
                    jnp.asarray(1.0 if greedy else temperature),
                    greedy,
                ),
                checkpoint_sha256=metadata["checkpoint_sha256"],
                weights_sha256=weights_digest,
            )
        )

    config = TrainingConfig.from_toml(args.config)
    environment = make_environment(
        config, config.curriculum[-1], pool_size=max(args.maps, 16)
    )
    pool, _ = environment.reset(jax.random.PRNGKey(args.map_seed))
    pool = jax.tree.map(lambda value: value[: args.maps], pool)
    pool = pool._replace(pool_idx=jnp.arange(args.maps, dtype=jnp.int32))

    @eqx.filter_jit
    def evaluate(pool_shard, network_a, network_b, key):
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
            sampling="categorical",
            key=key,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches = []
    started = time.perf_counter()
    base = {
        "schema": "single_checkpoint_temperature_round_robin_v1",
        "checkpoint": args.checkpoint,
        "map_seed": args.map_seed,
        "action_seed": args.action_seed,
        "maps_per_matchup": args.maps,
        "games_per_matchup": 2 * args.maps,
        "shard_maps": args.shard_maps,
        "locked_maps_across_matchups": True,
        "seat_swapped": True,
        "checkpoint_policy": metadata["policy"],
        "temperatures": sorted(args.temperatures),
        "greedy_encoded_as_temperature_zero": True,
    }
    write_payload(args.output, base, participants, matches, started)

    match_index = 0
    for index, first in enumerate(participants):
        for second in participants[index + 1 :]:
            match_started = time.perf_counter()
            shards = []
            for start in range(0, args.maps, args.shard_maps):
                pool_shard = jax.tree.map(
                    lambda value: value[start : start + args.shard_maps], pool
                )
                pool_shard = pool_shard._replace(
                    pool_idx=jnp.arange(args.shard_maps, dtype=jnp.int32)
                )
                key = jax.random.fold_in(
                    jax.random.PRNGKey(args.action_seed + match_index), start
                )
                shards.append(
                    scalar_result(
                        evaluate(pool_shard, first.network, second.network, key)
                    )
                )
            result = combine_shards(shards, args.maps)
            result.update(
                {
                    "a": first.name,
                    "b": second.name,
                    "evaluation_seconds": time.perf_counter() - match_started,
                }
            )
            matches.append(result)
            payload = write_payload(
                args.output, base, participants, matches, started
            )
            print(
                f"[{len(matches):2d}/{payload['total_matchups']}] "
                f"{first.name} vs {second.name}: "
                f"{int(result['wins'])}W/{int(result['losses'])}L/"
                f"{int(result['draws'])}D score={result['score']:.4f} "
                f"CI=[{result['score_ci95'][0]:.4f},{result['score_ci95'][1]:.4f}] "
                f"seconds={result['evaluation_seconds']:.1f}",
                flush=True,
            )
            match_index += 1

    payload = write_payload(args.output, base, participants, matches, started)
    print("FINAL RANKING", flush=True)
    for rank, row in enumerate(payload["ranking"], 1):
        print(
            f"{rank:2d}. {row['name']}: macro={row['macro_score']:.6f} "
            f"W/L/D={int(row['wins'])}/{int(row['losses'])}/"
            f"{int(row['draws'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()

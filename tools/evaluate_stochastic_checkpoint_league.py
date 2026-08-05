"""Paired stochastic round robin for heterogeneous training checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from generals.training.config import TrainingConfig
from generals.training.evaluate_head_to_head import _load_policy
from generals.training.evaluation import evaluate_paired_networks, round_robin_matrices
from generals.training.train import make_environment


@dataclass(frozen=True)
class Participant:
    name: str
    config: Path
    checkpoint: Path
    iteration: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> list[Participant]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    participants = [
        Participant(
            name=item["name"],
            config=Path(item["config"]),
            checkpoint=Path(item["checkpoint"]),
            iteration=int(item["iteration"]),
        )
        for item in payload["participants"]
    ]
    if len(participants) != 5:
        raise ValueError(f"Expected exactly five participants, got {len(participants)}")
    if len({item.name for item in participants}) != len(participants):
        raise ValueError("Participant names must be unique")
    return participants


def _sum_shards(shards: list[dict[str, np.ndarray]], maps: int) -> dict[str, float]:
    counts = {}
    for key in shards[0]:
        if key in {"score", "paired_score_std"}:
            continue
        counts[key] = float(sum(float(shard[key]) for shard in shards))
    games = counts["wins"] + counts["losses"] + counts["draws"]
    score = (counts["wins"] + 0.5 * counts["draws"]) / games
    shard_maps = maps // len(shards)
    second_moment = (
        sum(shard_maps * (float(shard["paired_score_std"]) ** 2 + float(shard["score"]) ** 2) for shard in shards)
        / maps
    )
    paired_std = math.sqrt(max(0.0, second_moment - score**2))
    return {
        **counts,
        "games": games,
        "score": score,
        "win_rate": counts["wins"] / games,
        "paired_score_std": paired_std,
        "score_ci95": [
            max(0.0, score - 1.96 * paired_std / math.sqrt(maps)),
            min(1.0, score + 1.96 * paired_std / math.sqrt(maps)),
        ],
    }


def run(args: argparse.Namespace) -> dict:
    participants = _load_manifest(args.manifest)
    configs = {item.name: TrainingConfig.from_toml(item.config) for item in participants}
    base = configs[participants[0].name]
    compatibility = (
        "pad_to",
        "history_size",
        "temporal_window",
        "observation_schema",
        "truncation",
        "deathtouch_turn",
    )
    for item in participants[1:]:
        config = configs[item.name]
        mismatches = [name for name in compatibility if getattr(config, name) != getattr(base, name)]
        if mismatches:
            raise ValueError(f"{item.name} has incompatible settings: {mismatches}")
    if args.maps % args.shard_maps:
        raise ValueError("maps must be divisible by shard-maps")

    # The final curriculum is a mixture of map-size/terrain combinations.  A
    # tiny smoke run still needs at least one board from every combination.
    environment = make_environment(base, base.curriculum[-1], pool_size=max(args.maps, 16))
    pool, _ = environment.reset(jax.random.PRNGKey(args.seed))
    pool = jax.tree.map(lambda value: value[: args.maps], pool)
    pool = pool._replace(pool_idx=jnp.arange(args.maps, dtype=jnp.int32))

    loaded = {}
    metadata = []
    for item in participants:
        network, iteration, stage = _load_policy(configs[item.name], item.checkpoint, "raw")
        if iteration != item.iteration:
            raise ValueError(f"{item.name} checkpoint iteration {iteration}, expected {item.iteration}")
        loaded[item.name] = jax.device_get(network)
        metadata.append(
            {
                "name": item.name,
                "iteration": iteration,
                "stage": stage,
                "config": str(item.config),
                "checkpoint": str(item.checkpoint),
                "checkpoint_sha256": _sha256(item.checkpoint),
                "policy": "raw",
            }
        )
        del network

    @eqx.filter_jit
    def evaluate(pool_shard, network_a, network_b, key):
        return evaluate_paired_networks(
            environment,
            pool_shard,
            network_a,
            network_b,
            args.shard_maps,
            base.truncation,
            schema_a=base.observation_schema,
            schema_b=base.observation_schema,
            pad_to=base.pad_to,
            history_size=base.history_size,
            temporal_window=base.temporal_window,
            sampling="categorical",
            key=key,
        )

    matches = []
    match_index = 0
    for i, first in enumerate(participants):
        for second in participants[i:]:
            started = time.perf_counter()
            shards = []
            for start in range(0, args.maps, args.shard_maps):
                shard = jax.tree.map(lambda value: value[start : start + args.shard_maps], pool)
                shard = shard._replace(pool_idx=jnp.arange(args.shard_maps, dtype=jnp.int32))
                key = jax.random.fold_in(jax.random.PRNGKey(args.action_seed + match_index), start)
                result = evaluate(shard, loaded[first.name], loaded[second.name], key)
                shards.append({name: np.asarray(jax.device_get(value)) for name, value in result.items()})
            combined = _sum_shards(shards, args.maps)
            combined.update(
                {
                    "a": first.name,
                    "b": second.name,
                    "sampling": "categorical",
                    "temperature": 1.0,
                    "maps": args.maps,
                    "evaluation_seconds": time.perf_counter() - started,
                }
            )
            for prefix in ("behavior_a_", "behavior_b_"):
                if combined[f"{prefix}builds"] != combined[f"{prefix}successful_builds"]:
                    raise RuntimeError(
                        f"{first.name} vs {second.name}: {prefix} legal build "
                        "actions did not equal successful constructions"
                    )
            matches.append(combined)
            print(
                f"{first.name} vs {second.name}: "
                f"{int(combined['wins'])}W/{int(combined['losses'])}L/"
                f"{int(combined['draws'])}D score={combined['score']:.4f} "
                f"castles={int(combined['behavior_a_successful_builds'])}/"
                f"{int(combined['behavior_b_successful_builds'])}",
                flush=True,
            )
            match_index += 1

    names = [item.name for item in participants]
    payload = {
        "seed": args.seed,
        "action_seed": args.action_seed,
        "sampling": "categorical",
        "temperature": 1.0,
        "maps": args.maps,
        "games_per_matchup": 2 * args.maps,
        "total_games": len(matches) * 2 * args.maps,
        "participants": metadata,
        "matches": matches,
        "matrices": round_robin_matrices(names, matches),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps", type=int, default=256)
    parser.add_argument("--shard-maps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--action-seed", type=int, default=20260807)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Round-robin evaluation of every archived checkpoint in a two-arm A/B."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import jax

from .config import TrainingConfig
from .evaluate_head_to_head import _combine, _load_policy, evaluate_paired_networks
from .tracking import WandbTracker
from .train import _shard_league_pool, make_environment


@dataclass(frozen=True)
class Participant:
    name: str
    arm: str
    iteration: int
    config_path: Path
    checkpoint_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _participants(
    control_config_path: Path,
    treatment_config_path: Path,
    parent_checkpoint: Path,
    iterations: tuple[int, ...],
) -> list[Participant]:
    control = TrainingConfig.from_toml(control_config_path)
    treatment = TrainingConfig.from_toml(treatment_config_path)
    participants = [
        Participant(
            "parent_003003",
            "parent",
            3003,
            control_config_path,
            parent_checkpoint,
        )
    ]
    for arm, config_path, config in (
        ("control_lambda097", control_config_path, control),
        ("treatment_phi_boost", treatment_config_path, treatment),
    ):
        for iteration in iterations:
            checkpoint = (
                config.run_dir / f"checkpoint_{iteration:06d}.eqx"
                if iteration < config.num_iterations
                else config.run_dir / "terminal.eqx"
            )
            participants.append(
                Participant(
                    f"{arm}_{iteration:06d}",
                    arm,
                    iteration,
                    config_path,
                    checkpoint,
                )
            )
    missing = [str(item.checkpoint_path) for item in participants if not item.checkpoint_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing round-robin checkpoints: {missing}")
    return participants


def run(args: argparse.Namespace) -> dict:
    participants = _participants(
        args.control_config,
        args.treatment_config,
        args.parent_checkpoint,
        tuple(args.iterations),
    )
    base = TrainingConfig.from_toml(args.control_config)
    tracker_config = replace(
        base,
        run_name="castle_ab_lambda097_checkpoint_round_robin_20260804",
        wandb_run_id="castle-ab-lambda097-checkpoint-round-robin-20260804",
        wandb_run_name="Castle A/B · all-checkpoint round robin · λ=0.97",
        wandb_job_type="evaluation-round-robin",
        wandb_tags=(
            "castle-ab",
            "lambda097",
            "checkpoint-round-robin",
            "raw-and-ema",
        ),
    )
    tracker = WandbTracker.start(
        tracker_config,
        start_iteration=0,
        resume=None,
        device_count=jax.device_count(),
    )
    environment = make_environment(
        base,
        base.curriculum[-1],
        pool_size=max(args.maps, 16),
    )
    pool, _ = environment.reset(jax.random.PRNGKey(args.seed))
    device_count = jax.device_count()
    if args.maps % device_count:
        raise ValueError(
            f"maps ({args.maps}) must be divisible by devices ({device_count})"
        )
    sharded_pool = _shard_league_pool(pool, args.maps, device_count)
    maps_per_device = args.maps // device_count
    evaluator = jax.pmap(
        lambda pool_shard, network_a, network_b: evaluate_paired_networks(
            environment,
            pool_shard,
            network_a,
            network_b,
            maps_per_device,
            base.truncation,
            schema_a=base.observation_schema,
            schema_b=base.observation_schema,
            pad_to=base.pad_to,
            history_size=base.history_size,
            temporal_window=base.temporal_window,
        ),
        in_axes=(0, None, None),
    )

    payload: dict[str, object] = {
        "seed": args.seed,
        "maps": args.maps,
        "games_per_matchup": 2 * args.maps,
        "participants": [
            {
                "name": item.name,
                "arm": item.arm,
                "iteration": item.iteration,
                "checkpoint": str(item.checkpoint_path),
                "checkpoint_sha256": _sha256(item.checkpoint_path),
            }
            for item in participants
        ],
        "policies": {},
    }
    match_index = 0
    for policy in args.policies:
        loaded = {}
        for item in participants:
            config = TrainingConfig.from_toml(item.config_path)
            network, loaded_iteration, _ = _load_policy(
                config, item.checkpoint_path, policy
            )
            if loaded_iteration != item.iteration:
                raise ValueError(
                    f"{item.name} contains iteration {loaded_iteration}, expected {item.iteration}"
                )
            loaded[item.name] = jax.device_get(network)

        standings = {
            item.name: {"score_sum": 0.0, "games": 0.0, "matchups": 0}
            for item in participants
        }
        matchups = []
        for first, second in itertools.combinations(participants, 2):
            started = time.perf_counter()
            metrics = evaluator(
                sharded_pool,
                loaded[first.name],
                loaded[second.name],
            )
            result = _combine(jax.device_get(metrics))
            result["evaluation_seconds"] = time.perf_counter() - started
            result.update({"a": first.name, "b": second.name})
            matchups.append(result)
            games = result["games"]
            standings[first.name]["score_sum"] += result["score"] * games
            standings[first.name]["games"] += games
            standings[first.name]["matchups"] += 1
            standings[second.name]["score_sum"] += (1.0 - result["score"]) * games
            standings[second.name]["games"] += games
            standings[second.name]["matchups"] += 1
            tracker.log_evaluation(
                {
                    "iteration": match_index,
                    "round_robin/policy": policy,
                    "round_robin/a": first.name,
                    "round_robin/b": second.name,
                    "round_robin/a_score": result["score"],
                    "round_robin/games": games,
                }
            )
            match_index += 1
            print(
                f"{policy}: {first.name} vs {second.name}: "
                f"{int(result['wins'])}W/{int(result['losses'])}L/"
                f"{int(result['draws'])}D, score={result['score']:.3f}",
                flush=True,
            )
        for record in standings.values():
            record["score"] = record.pop("score_sum") / max(record["games"], 1.0)
        ranking = sorted(
            (
                {"participant": name, **record}
                for name, record in standings.items()
            ),
            key=lambda value: value["score"],
            reverse=True,
        )
        payload["policies"][policy] = {
            "matchups": matchups,
            "standings": standings,
            "ranking": ranking,
        }
        del loaded

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    tracker.update_summary(
        {
            "participants": len(participants),
            "matchups_per_policy": len(participants) * (len(participants) - 1) // 2,
            "output": str(args.output),
        }
    )
    tracker.finish()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--treatment-config", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2044)
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=(3200, 3400, 3600, 3800, 4000, 4003),
    )
    parser.add_argument("--policies", nargs="+", choices=("raw", "ema"), default=("raw", "ema"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

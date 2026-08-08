"""Greedy paired-map round robin for raw and EMA policies from checkpoints 14k--19k."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


ITERATIONS = (14000, 15000, 16000, 17000, 18000, 19000)
CHECKPOINT_SHA256 = {
    14000: "276413ebbe5a180cf5c699e45d48878375ac0aeba954726ab43b40ae75b93543",
    15000: "49ae3fc5841333ad09a6fd0bd68ae3ba3f0aedd8ea08dd56fc294d92e3fa91a3",
    16000: "63a2428bade0800bc0b01161fe785e8e5b64391feb37e29c4972904cf5f64ade",
    17000: "000ebcd3c1a5eae614bc239715c46802e84a904f93dd40a94f8224dce0261303",
    18000: "44c5ec72bc3fa9954597eb0a6ae8cee2993cd479e29d94c15b4766a112a7cf85",
    19000: "5cb1ee6ec929f9c144751f51639baa850a4f7fb79febd8351157daac9ad0f2ae",
}


@dataclass(frozen=True)
class Participant:
    name: str
    iteration: int
    policy: str
    checkpoint: Path
    checkpoint_sha256: str
    stage: int
    network: object

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "iteration": self.iteration,
            "policy": self.policy,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "curriculum_stage": self.stage,
            "inference": "greedy",
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_result(result) -> dict[str, float]:
    return {
        key: float(np.asarray(jax.device_get(value)))
        for key, value in result.items()
    }


def combine_shards(shards: list[dict[str, float]], maps: int) -> dict:
    counts = {}
    for key in shards[0]:
        if key in {"score", "paired_score_std"}:
            continue
        counts[key] = float(sum(shard[key] for shard in shards))
    games = counts["wins"] + counts["losses"] + counts["draws"]
    score = (counts["wins"] + 0.5 * counts["draws"]) / games
    shard_maps = maps // len(shards)
    second_moment = sum(
        shard_maps
        * (shard["paired_score_std"] ** 2 + shard["score"] ** 2)
        for shard in shards
    ) / maps
    paired_std = math.sqrt(max(0.0, second_moment - score**2))
    half_width = 1.96 * paired_std / math.sqrt(maps)
    return {
        **counts,
        "games": games,
        "score": score,
        "paired_score_std": paired_std,
        "score_ci95": [
            max(0.0, score - half_width),
            min(1.0, score + half_width),
        ],
    }


def summaries(names: list[str], matches: list[dict]) -> dict[str, dict]:
    rows = {}
    for name in names:
        scores = []
        wins = losses = draws = 0.0
        for match in matches:
            if match["a"] == name:
                scores.append(match["score"])
                wins += match["wins"]
                losses += match["losses"]
                draws += match["draws"]
            elif match["b"] == name:
                scores.append(1.0 - match["score"])
                wins += match["losses"]
                losses += match["wins"]
                draws += match["draws"]
        games = wins + losses + draws
        rows[name] = {
            "completed_opponents": len(scores),
            "macro_score": sum(scores) / len(scores) if scores else None,
            "micro_score": (wins + 0.5 * draws) / games if games else None,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "games": games,
        }
    return rows


def matrices(names: list[str], matches: list[dict]) -> dict:
    index = {name: i for i, name in enumerate(names)}
    score = [[None for _ in names] for _ in names]
    for i in range(len(names)):
        score[i][i] = 0.5
    for match in matches:
        i, j = index[match["a"]], index[match["b"]]
        score[i][j] = match["score"]
        score[j][i] = 1.0 - match["score"]
    return {"labels": names, "score": score}


def write_payload(
    path: Path,
    base: dict,
    participants: list[Participant],
    matches: list[dict],
    started: float,
) -> dict:
    names = [item.name for item in participants]
    summary = summaries(names, matches)
    complete = all(
        row["completed_opponents"] == len(names) - 1 for row in summary.values()
    )
    ranking = sorted(
        (
            {"name": name, **row}
            for name, row in summary.items()
            if row["macro_score"] is not None
        ),
        key=lambda row: row["macro_score"],
        reverse=True,
    )
    payload = {
        **base,
        "complete": complete,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_matchups": len(matches),
        "total_matchups": len(names) * (len(names) - 1) // 2,
        "total_games": int(sum(match["games"] for match in matches)),
        "participants": [item.metadata() for item in participants],
        "matches": matches,
        "summaries": summary,
        "ranking": ranking,
        "matrices": matrices(names, matches),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return payload


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
    participants = []
    for iteration in ITERATIONS:
        checkpoint = args.checkpoint_root / f"checkpoint_{iteration:06d}.eqx"
        digest = sha256(checkpoint)
        if digest != CHECKPOINT_SHA256[iteration]:
            raise RuntimeError(
                f"checkpoint hash mismatch for {iteration}: {digest}"
            )
        for policy in ("raw", "ema"):
            network, loaded_iteration, stage = _load_policy(
                config, checkpoint, policy
            )
            if loaded_iteration != iteration:
                raise RuntimeError(
                    f"checkpoint {iteration} contains iteration {loaded_iteration}"
                )
            participants.append(
                Participant(
                    name=f"c{iteration}_{policy}",
                    iteration=iteration,
                    policy=policy,
                    checkpoint=checkpoint,
                    checkpoint_sha256=digest,
                    stage=stage,
                    network=network,
                )
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches = []
    if args.output.is_file():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("map_seed") != args.map_seed:
            raise RuntimeError("refusing to resume with a different map seed")
        matches = prior.get("matches", [])
    completed = {(match["a"], match["b"]) for match in matches}
    started = time.perf_counter()
    base = {
        "schema": "checkpoint_raw_ema_greedy_round_robin_v1",
        "map_seed": args.map_seed,
        "maps_per_matchup": args.maps,
        "games_per_matchup": 2 * args.maps,
        "shard_maps": args.shard_maps,
        "locked_maps_across_matchups": True,
        "seat_swapped": True,
        "inference": "greedy",
        "iterations": list(ITERATIONS),
        "policies": ["raw", "ema"],
    }
    write_payload(args.output, base, participants, matches, started)

    pairs = list(itertools.combinations(participants, 2))
    for match_index, (first, second) in enumerate(pairs):
        pair = (first.name, second.name)
        if pair in completed:
            continue
        match_started = time.perf_counter()
        shards = []
        for start in range(0, args.maps, args.shard_maps):
            pool_shard = jax.tree.map(
                lambda value: value[start : start + args.shard_maps], pool
            )
            pool_shard = pool_shard._replace(
                pool_idx=jnp.arange(args.shard_maps, dtype=jnp.int32)
            )
            shards.append(
                scalar_result(evaluate(pool_shard, first.network, second.network))
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
        payload = write_payload(args.output, base, participants, matches, started)
        leader = payload["ranking"][0]
        print(
            f"[{len(matches):2d}/{payload['total_matchups']}] "
            f"{first.name} vs {second.name}: "
            f"{int(result['wins'])}W/{int(result['losses'])}L/"
            f"{int(result['draws'])}D score={result['score']:.4f} "
            f"CI=[{result['score_ci95'][0]:.4f},{result['score_ci95'][1]:.4f}] "
            f"seconds={result['evaluation_seconds']:.1f} "
            f"leader={leader['name']}({leader['macro_score']:.4f})",
            flush=True,
        )

    payload = write_payload(args.output, base, participants, matches, started)
    print("FINAL RANKING", flush=True)
    for rank, row in enumerate(payload["ranking"], 1):
        print(
            f"{rank:2d}. {row['name']}: macro={row['macro_score']:.6f} "
            f"micro={row['micro_score']:.6f} "
            f"W/L/D={int(row['wins'])}/{int(row['losses'])}/{int(row['draws'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()

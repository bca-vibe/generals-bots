from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


TEMPERATURES = (None, 0.25, 0.5, 0.75, 1.0)


class StrategyNetwork(eqx.Module):
    parameters: dict
    policy_logits: object = eqx.field(static=True)
    temperature: jax.Array
    greedy_proxy: bool = eqx.field(static=True)

    def forward(self, observation, temporal_history, legal_mask):
        logits = self.policy_logits(
            self.parameters, observation, temporal_history, legal_mask
        ).astype(jnp.float32)
        if self.greedy_proxy:
            chosen = jnp.argmax(logits)
            logits = jnp.where(
                jnp.arange(logits.shape[0]) == chosen, 0.0, -1e9
            )
        else:
            logits = logits / self.temperature
        return logits, jnp.zeros((), dtype=jnp.float32)


@dataclass(frozen=True)
class Participant:
    name: str
    checkpoint: int
    temperature: float | None
    network: StrategyNetwork
    checkpoint_sha256: str
    weights_sha256: str

    def metadata(self):
        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "inference": "greedy" if self.temperature is None else "categorical",
            "temperature": self.temperature,
            "checkpoint_sha256": self.checkpoint_sha256,
            "weights_sha256": self.weights_sha256,
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bot(path: Path):
    spec = importlib.util.spec_from_file_location("submission_bot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_parameters(path: Path):
    with np.load(path, allow_pickle=False) as stored:
        host = {
            name: jnp.asarray(stored[name]).view(jnp.bfloat16)
            for name in stored.files
        }
    return jax.device_put(host)


def scalar_result(result):
    return {
        key: float(np.asarray(jax.device_get(value)))
        for key, value in result.items()
    }


def combine_shards(shards, maps):
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


def summaries(names, matches):
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


def matrices(names, matches):
    index = {name: i for i, name in enumerate(names)}
    score = [[None for _ in names] for _ in names]
    for i in range(len(names)):
        score[i][i] = 0.5
    for match in matches:
        i, j = index[match["a"]], index[match["b"]]
        score[i][j] = match["score"]
        score[j][i] = 1.0 - match["score"]
    return {"labels": names, "score": score}


def write_payload(path, base, participants, matches, started):
    names = [item.name for item in participants]
    summary = summaries(names, matches)
    complete = all(row["completed_opponents"] == len(names) - 1 for row in summary.values())
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
        "participants": [item.metadata() for item in participants],
        "matches": matches,
        "summaries": summary,
        "ranking": ranking,
        "matrices": matrices(names, matches),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--agent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps", type=int, default=512)
    parser.add_argument("--shard-maps", type=int, default=128)
    parser.add_argument("--map-seed", type=int, default=202608071)
    parser.add_argument("--action-seed", type=int, default=202608072)
    args = parser.parse_args()
    if args.maps % args.shard_maps:
        raise ValueError("maps must be divisible by shard-maps")

    import sys

    sys.path.insert(0, str(args.repo))
    from generals.training.config import TrainingConfig
    from generals.training.evaluation import evaluate_paired_networks
    from generals.training.train import make_environment

    bot = load_bot(args.agent_root / "c14000" / "bot.py")
    policy_logits = bot._policy_logits
    participants = []
    for checkpoint in (10000, 12000, 14000):
        directory = args.agent_root / f"c{checkpoint}"
        metadata = json.loads((directory / "export_metadata.json").read_text())
        weights_path = directory / "weights.npz"
        weights_digest = sha256(weights_path)
        if weights_digest != metadata["weights_sha256"]:
            raise RuntimeError(f"weights hash mismatch for {checkpoint}")
        parameters = load_parameters(weights_path)
        for temperature in TEMPERATURES:
            suffix = "greedy" if temperature is None else f"t{temperature:g}"
            participants.append(
                Participant(
                    name=f"c{checkpoint}_{suffix}",
                    checkpoint=checkpoint,
                    temperature=temperature,
                    network=StrategyNetwork(
                        parameters,
                        policy_logits,
                        jnp.asarray(1.0 if temperature is None else temperature),
                        temperature is None,
                    ),
                    checkpoint_sha256=metadata["checkpoint_sha256"],
                    weights_sha256=weights_digest,
                )
            )

    config = TrainingConfig.from_toml(
        args.repo
        / "runs/castle_ppo_runpod_4xh100_from_012200_20260807/recovery_artifacts/castle_ppo_runpod_4xh100_from_12200.toml"
    )
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
    if args.output.is_file():
        prior = json.loads(args.output.read_text())
        if prior.get("map_seed") != args.map_seed or prior.get("action_seed") != args.action_seed:
            raise RuntimeError("refusing to resume with different seeds")
        matches = prior.get("matches", [])
    completed = {(match["a"], match["b"]) for match in matches}
    started = time.perf_counter()
    base = {
        "schema": "checkpoint_temperature_round_robin_v1",
        "map_seed": args.map_seed,
        "action_seed": args.action_seed,
        "maps_per_matchup": args.maps,
        "games_per_matchup": 2 * args.maps,
        "shard_maps": args.shard_maps,
        "locked_maps_across_matchups": True,
        "seat_swapped": True,
        "checkpoint_policy": "raw",
        "temperatures": list(TEMPERATURES[1:]),
    }
    write_payload(args.output, base, participants, matches, started)

    match_index = 0
    for i, first in enumerate(participants):
        for second in participants[i + 1 :]:
            pair = (first.name, second.name)
            if pair in completed:
                match_index += 1
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
                key = jax.random.fold_in(
                    jax.random.PRNGKey(args.action_seed + match_index), start
                )
                shards.append(
                    scalar_result(
                        evaluate(
                            pool_shard,
                            first.network,
                            second.network,
                            key,
                        )
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
            leader = payload["ranking"][0]
            print(
                f"[{len(matches):3d}/{payload['total_matchups']}] "
                f"{first.name} vs {second.name}: "
                f"{int(result['wins'])}W/{int(result['losses'])}L/"
                f"{int(result['draws'])}D score={result['score']:.4f} "
                f"CI=[{result['score_ci95'][0]:.4f},{result['score_ci95'][1]:.4f}] "
                f"seconds={result['evaluation_seconds']:.1f} "
                f"current_leader={leader['name']}({leader['macro_score']:.4f})",
                flush=True,
            )
            match_index += 1

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

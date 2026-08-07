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


def select_bracket(coarse):
    winner = coarse["ranking"][0]
    winner_meta = next(
        row for row in coarse["participants"] if row["name"] == winner["name"]
    )
    checkpoint = int(winner_meta["checkpoint"])
    summary = coarse["summaries"]
    settings = []
    for participant in coarse["participants"]:
        if int(participant["checkpoint"]) != checkpoint:
            continue
        temperature = 0.0 if participant["temperature"] is None else float(participant["temperature"])
        settings.append((temperature, participant["name"], summary[participant["name"]]["macro_score"]))
    settings.sort()
    best_index = max(range(len(settings)), key=lambda index: settings[index][2])
    neighbors = [index for index in (best_index - 1, best_index + 1) if 0 <= index < len(settings)]
    neighbor_index = max(neighbors, key=lambda index: settings[index][2])
    lo, hi = sorted((settings[best_index][0], settings[neighbor_index][0]))
    temperatures = []
    current = round(lo, 10)
    while current <= hi + 1e-9:
        temperatures.append(round(current, 10))
        current = round(current + 0.05, 10)
    return checkpoint, settings[best_index], settings[neighbor_index], temperatures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--agent-root", type=Path, required=True)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps", type=int, default=512)
    parser.add_argument("--shard-maps", type=int, default=128)
    parser.add_argument("--map-seed", type=int, default=202608073)
    parser.add_argument("--action-seed", type=int, default=202608074)
    args = parser.parse_args()
    if args.maps % args.shard_maps:
        raise ValueError("maps must be divisible by shard-maps")

    import sys

    sys.path.insert(0, str(args.repo))
    from generals.training.config import TrainingConfig
    from generals.training.evaluation import evaluate_paired_networks
    from generals.training.train import make_environment

    coarse = json.loads(args.coarse.read_text())
    if not coarse.get("complete"):
        raise RuntimeError("coarse round robin is incomplete")
    checkpoint, coarse_best, coarse_neighbor, temperatures = select_bracket(coarse)
    print(
        f"selected checkpoint={checkpoint} bracket={temperatures[0]:g}..{temperatures[-1]:g} "
        f"from best={coarse_best} neighbor={coarse_neighbor}",
        flush=True,
    )

    bot = load_bot(args.agent_root / "c14000" / "bot.py")
    directory = args.agent_root / f"c{checkpoint}"
    metadata = json.loads((directory / "export_metadata.json").read_text())
    weights_path = directory / "weights.npz"
    weights_digest = sha256(weights_path)
    if weights_digest != metadata["weights_sha256"]:
        raise RuntimeError("weights hash mismatch")
    parameters = load_parameters(weights_path)
    participants = []
    for temperature in temperatures:
        greedy = temperature == 0.0
        suffix = "greedy" if greedy else f"t{temperature:g}"
        participants.append(
            Participant(
                name=f"c{checkpoint}_{suffix}",
                checkpoint=checkpoint,
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

    config = TrainingConfig.from_toml(
        args.repo
        / "runs/castle_ppo_runpod_4xh100_from_012200_20260807/recovery_artifacts/castle_ppo_runpod_4xh100_from_12200.toml"
    )
    environment = make_environment(config, config.curriculum[-1], pool_size=max(args.maps, 16))
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
        "schema": "fine_temperature_round_robin_v1",
        "coarse_result": str(args.coarse),
        "coarse_winner": coarse["ranking"][0]["name"],
        "coarse_best_setting": {
            "temperature": coarse_best[0],
            "name": coarse_best[1],
            "macro_score": coarse_best[2],
        },
        "coarse_neighbor_setting": {
            "temperature": coarse_neighbor[0],
            "name": coarse_neighbor[1],
            "macro_score": coarse_neighbor[2],
        },
        "checkpoint": checkpoint,
        "map_seed": args.map_seed,
        "action_seed": args.action_seed,
        "maps_per_matchup": args.maps,
        "games_per_matchup": 2 * args.maps,
        "shard_maps": args.shard_maps,
        "locked_maps_across_matchups": True,
        "fresh_maps_relative_to_coarse": True,
        "seat_swapped": True,
        "checkpoint_policy": "raw",
        "temperatures": temperatures,
        "temperature_step": 0.05,
        "greedy_encoded_as_temperature_zero": True,
    }
    write_payload(args.output, base, participants, matches, started)

    match_index = 0
    for i, first in enumerate(participants):
        for second in participants[i + 1 :]:
            match_started = time.perf_counter()
            shards = []
            for start in range(0, args.maps, args.shard_maps):
                pool_shard = jax.tree.map(lambda value: value[start : start + args.shard_maps], pool)
                pool_shard = pool_shard._replace(pool_idx=jnp.arange(args.shard_maps, dtype=jnp.int32))
                key = jax.random.fold_in(jax.random.PRNGKey(args.action_seed + match_index), start)
                shards.append(scalar_result(evaluate(pool_shard, first.network, second.network, key)))
            result = combine_shards(shards, args.maps)
            result.update({"a": first.name, "b": second.name, "evaluation_seconds": time.perf_counter() - match_started})
            matches.append(result)
            payload = write_payload(args.output, base, participants, matches, started)
            print(
                f"[{len(matches):2d}/{payload['total_matchups']}] {first.name} vs {second.name}: "
                f"{int(result['wins'])}W/{int(result['losses'])}L/{int(result['draws'])}D "
                f"score={result['score']:.4f} CI=[{result['score_ci95'][0]:.4f},{result['score_ci95'][1]:.4f}] "
                f"seconds={result['evaluation_seconds']:.1f}",
                flush=True,
            )
            match_index += 1

    payload = write_payload(args.output, base, participants, matches, started)
    print("FINAL FINE RANKING", flush=True)
    for rank, row in enumerate(payload["ranking"], 1):
        print(f"{rank:2d}. {row['name']}: macro={row['macro_score']:.6f} W/L/D={int(row['wins'])}/{int(row['losses'])}/{int(row['draws'])}", flush=True)


if __name__ == "__main__":
    main()

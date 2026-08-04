"""Evaluate new candidate policies directly against a historical checkpoint.

The two sides may use different observation schemas.  This is required for
comparing current ``competition_39`` policies with the original
``legacy_38`` transformer while keeping the played maps and rules identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from generals.core.game import get_observation

from .actions import CELL_COUNT, MOVE_PLANES, PASS_INDEX, decode_action, legal_action_mask
from .config import TrainingConfig
from .evaluation import _batched_memory
from .observation import augment_observation, temporal_input
from .train import (
    _learning_rate,
    _load_checkpoint_state,
    _shard_league_pool,
    build_network,
    make_environment,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy(config: TrainingConfig, path: Path, policy: str):
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
    raw, _, ema, iteration, stage, _ = _load_checkpoint_state(path, skeleton, config)
    selected = ema if policy == "ema" else raw
    return selected, int(iteration), int(stage)


def evaluate_paired_networks(
    environment,
    pool,
    network_a,
    network_b,
    n_maps: int,
    truncation: int,
    *,
    schema_a: str,
    schema_b: str,
    pad_to: int = 21,
    history_size: int = 7,
    temporal_window: int = 512,
):
    """Play every map twice and report results from network A's perspective."""
    selected = jax.tree.map(lambda value: value[:n_maps], pool)
    states = jax.tree.map(lambda value: jnp.concatenate([value, value]), selected)
    games = 2 * n_maps
    a_is_zero = jnp.arange(games) < n_maps
    memory_a = _batched_memory(games, pad_to, history_size, temporal_window)
    memory_b = _batched_memory(games, pad_to, history_size, temporal_window)
    finished = jnp.zeros((games,), dtype=jnp.bool_)
    outcomes = jnp.full((games,), 0.5, dtype=jnp.float32)
    first_build_a = jnp.full((games,), -1, dtype=jnp.int32)
    first_build_b = jnp.full((games,), -1, dtype=jnp.int32)
    zero_count = jnp.zeros((), dtype=jnp.int32)

    def select_by_seat(observation_zero, observation_one, choose_zero):
        return jax.tree.map(
            lambda zero, one: jnp.where(
                choose_zero.reshape((-1,) + (1,) * (zero.ndim - 1)), zero, one
            ),
            observation_zero,
            observation_one,
        )

    def step(carry, _):
        (
            states,
            memory_a,
            memory_b,
            finished,
            outcomes,
            actions_a_count,
            actions_b_count,
            builds_a_count,
            builds_b_count,
            opportunities_a_count,
            opportunities_b_count,
            first_build_a,
            first_build_b,
        ) = carry
        observation_zero = jax.vmap(lambda state: get_observation(state, 0))(states)
        observation_one = jax.vmap(lambda state: get_observation(state, 1))(states)
        observation_a = select_by_seat(observation_zero, observation_one, a_is_zero)
        observation_b = select_by_seat(observation_zero, observation_one, ~a_is_zero)

        augmented_a, memory_a = jax.vmap(
            lambda observation, memory, board_mask: augment_observation(
                observation,
                memory,
                board_mask,
                schema_a,
                environment.deathtouch_turn or 800,
            )
        )(observation_a, memory_a, states.board_mask)
        augmented_b, memory_b = jax.vmap(
            lambda observation, memory, board_mask: augment_observation(
                observation,
                memory,
                board_mask,
                schema_b,
                environment.deathtouch_turn or 800,
            )
        )(observation_b, memory_b, states.board_mask)
        history_a = temporal_input(memory_a)
        history_b = temporal_input(memory_b)
        mask_a = jax.vmap(legal_action_mask)(observation_a, states.board_mask)
        mask_b = jax.vmap(legal_action_mask)(observation_b, states.board_mask)
        action_a = jax.vmap(
            lambda obs, history, mask: decode_action(
                jnp.argmax(network_a.forward(obs, history, mask)[0])
            )
        )(augmented_a, history_a, mask_a)
        action_b = jax.vmap(
            lambda obs, history, mask: decode_action(
                jnp.argmax(network_b.forward(obs, history, mask)[0])
            )
        )(augmented_b, history_b, mask_b)

        active = ~finished
        build_a = active & (action_a[:, 0] == 2)
        build_b = active & (action_b[:, 0] == 2)
        build_slice = slice(MOVE_PLANES * CELL_COUNT, PASS_INDEX)
        opportunity_a = active & jnp.any(mask_a[:, build_slice], axis=1)
        opportunity_b = active & jnp.any(mask_b[:, build_slice], axis=1)
        actions_a_count += active.sum()
        actions_b_count += active.sum()
        builds_a_count += build_a.sum()
        builds_b_count += build_b.sum()
        opportunities_a_count += opportunity_a.sum()
        opportunities_b_count += opportunity_b.sum()
        turn_a = observation_a.timestep.astype(jnp.int32)
        turn_b = observation_b.timestep.astype(jnp.int32)
        first_build_a = jnp.where(build_a & (first_build_a < 0), turn_a, first_build_a)
        first_build_b = jnp.where(build_b & (first_build_b < 0), turn_b, first_build_b)

        actions_zero = jnp.where(a_is_zero[:, None], action_a, action_b)
        actions_one = jnp.where(a_is_zero[:, None], action_b, action_a)
        timesteps, states = jax.vmap(
            lambda state, actions: environment.step(state, actions, pool)
        )(states, jnp.stack([actions_zero, actions_one], axis=1))
        done = timesteps.terminated | timesteps.truncated
        newly_finished = done & ~finished
        a_won = jnp.where(
            a_is_zero, timesteps.info.winner == 0, timesteps.info.winner == 1
        )
        a_lost = jnp.where(
            a_is_zero, timesteps.info.winner == 1, timesteps.info.winner == 0
        )
        result = jnp.where(a_won, 1.0, jnp.where(a_lost, 0.0, 0.5))
        outcomes = jnp.where(newly_finished, result, outcomes)
        finished |= done
        return (
            states,
            memory_a,
            memory_b,
            finished,
            outcomes,
            actions_a_count,
            actions_b_count,
            builds_a_count,
            builds_b_count,
            opportunities_a_count,
            opportunities_b_count,
            first_build_a,
            first_build_b,
        ), None

    initial = (
        states,
        memory_a,
        memory_b,
        finished,
        outcomes,
        zero_count,
        zero_count,
        zero_count,
        zero_count,
        zero_count,
        zero_count,
        first_build_a,
        first_build_b,
    )
    final, _ = jax.lax.scan(step, initial, None, length=truncation)
    (
        _,
        _,
        _,
        finished,
        outcomes,
        actions_a_count,
        actions_b_count,
        builds_a_count,
        builds_b_count,
        opportunities_a_count,
        opportunities_b_count,
        first_build_a,
        first_build_b,
    ) = final
    wins = jnp.sum((outcomes == 1.0) & finished)
    losses = jnp.sum((outcomes == 0.0) & finished)
    draws = games - wins - losses
    paired_scores = (outcomes[:n_maps] + outcomes[n_maps:]) / 2.0
    built_a = first_build_a >= 0
    built_b = first_build_b >= 0
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": outcomes.mean(),
        "paired_score_std": paired_scores.std(),
        "actions_a": actions_a_count,
        "actions_b": actions_b_count,
        "builds_a": builds_a_count,
        "builds_b": builds_b_count,
        "build_opportunities_a": opportunities_a_count,
        "build_opportunities_b": opportunities_b_count,
        "games_with_build_a": built_a.sum(),
        "games_with_build_b": built_b.sum(),
        "first_build_turn_sum_a": jnp.where(built_a, first_build_a, 0).sum(),
        "first_build_turn_sum_b": jnp.where(built_b, first_build_b, 0).sum(),
    }


def _combine(metrics) -> dict[str, float]:
    host = {name: np.asarray(value, dtype=np.float64) for name, value in metrics.items()}
    sums = {
        name: float(values.sum())
        for name, values in host.items()
        if name not in {"score", "paired_score_std"}
    }
    score = float(host["score"].mean())
    second_moment = np.mean(host["paired_score_std"] ** 2 + host["score"] ** 2)
    result = {
        **sums,
        "games": sums["wins"] + sums["losses"] + sums["draws"],
        "score": score,
        "paired_score_std": float(np.sqrt(max(0.0, second_moment - score**2))),
    }
    for side in ("a", "b"):
        actions = result[f"actions_{side}"]
        builds = result[f"builds_{side}"]
        opportunities = result[f"build_opportunities_{side}"]
        built_games = result[f"games_with_build_{side}"]
        result[f"build_action_share_{side}"] = builds / max(actions, 1.0)
        result[f"build_rate_when_legal_{side}"] = builds / max(opportunities, 1.0)
        result[f"builds_per_game_{side}"] = builds / max(result["games"], 1.0)
        result[f"games_with_build_rate_{side}"] = built_games / max(result["games"], 1.0)
        result[f"mean_first_build_turn_{side}"] = (
            result[f"first_build_turn_sum_{side}"] / max(built_games, 1.0)
        )
    return result


def run(args) -> dict:
    old_config = TrainingConfig.from_toml(args.original_config)
    original, original_iteration, original_stage = _load_policy(
        old_config, args.original_checkpoint, args.policy
    )
    candidate_specs = (
        ("transformer", args.transformer_config, args.transformer_checkpoint),
        ("convolutional", args.conv_config, args.conv_checkpoint),
    )
    candidate_records = []
    for name, config_path, checkpoint_path in candidate_specs:
        config = TrainingConfig.from_toml(config_path)
        network, iteration, stage = _load_policy(config, checkpoint_path, args.policy)
        candidate_records.append((name, config, checkpoint_path, network, iteration, stage))

    reference_config = candidate_records[0][1]
    environment = make_environment(
        reference_config,
        reference_config.curriculum[-1],
        pool_size=max(args.maps, 16),
    )
    pool_key = jax.random.PRNGKey(args.seed)
    pool, _ = environment.reset(pool_key)
    device_count = jax.device_count()
    if args.maps % device_count:
        raise ValueError(f"maps ({args.maps}) must be divisible by devices ({device_count})")
    sharded_pool = _shard_league_pool(pool, args.maps, device_count)
    maps_per_device = args.maps // device_count
    results = {}
    for name, config, checkpoint_path, network, iteration, stage in candidate_records:
        evaluator = jax.pmap(
            lambda pool_shard, candidate, historical: evaluate_paired_networks(
                environment,
                pool_shard,
                candidate,
                historical,
                maps_per_device,
                reference_config.truncation,
                schema_a=config.observation_schema,
                schema_b=old_config.observation_schema,
                pad_to=reference_config.pad_to,
                history_size=reference_config.history_size,
                temporal_window=reference_config.temporal_window,
            ),
            in_axes=(0, None, None),
        )
        started = time.perf_counter()
        metrics = evaluator(sharded_pool, jax.device_get(network), jax.device_get(original))
        result = _combine(jax.device_get(metrics))
        result.update(
            {
                "evaluation_seconds": time.perf_counter() - started,
                "candidate_iteration": iteration,
                "candidate_stage": stage,
                "candidate_checkpoint": str(checkpoint_path),
                "candidate_checkpoint_sha256": _sha256(checkpoint_path),
            }
        )
        results[name] = result
        print(
            f"{name} vs original-{original_iteration}: "
            f"{int(result['wins'])}W/{int(result['losses'])}L/{int(result['draws'])}D, "
            f"score={result['score']:.3f}",
            flush=True,
        )

    payload = {
        "policy": args.policy,
        "seed": args.seed,
        "maps": args.maps,
        "games_per_matchup": 2 * args.maps,
        "rules": "final_competition",
        "original_checkpoint": str(args.original_checkpoint),
        "original_checkpoint_sha256": _sha256(args.original_checkpoint),
        "original_iteration": original_iteration,
        "original_stage": original_stage,
        "original_observation_schema": old_config.observation_schema,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-config", required=True, type=Path)
    parser.add_argument("--original-checkpoint", required=True, type=Path)
    parser.add_argument("--transformer-config", required=True, type=Path)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--conv-config", required=True, type=Path)
    parser.add_argument("--conv-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1044)
    parser.add_argument("--policy", choices=("raw", "ema"), default="ema")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""Paired-map benchmark for one in-repository heuristic against the league."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from generals.core.env import GeneralsEnv
from generals.core.game import get_observation
from generals.training.league import make_opponent_policy

DEFAULT_OPPONENTS = (
    "random",
    "boss",
    "expander",
    "hunter",
    "harvester",
    "human_exe",
    "castle_economist",
    "deathtouch_clock",
    "draw_grinder",
    "fog_scout",
    "raider",
    "sentinel",
)


def make_evaluator(environment, pool, maps, truncation, candidate_name, opponent_name):
    candidate_policy = make_opponent_policy(candidate_name)
    opponent_policy = make_opponent_policy(opponent_name)
    selected = jax.tree.map(lambda value: value[:maps], pool)
    initial_states = jax.tree.map(
        lambda value: jnp.concatenate([value, value]), selected
    )
    candidate_is_zero = jnp.arange(2 * maps) < maps

    def batched_memory(policy):
        memory = policy.initial_memory(initial_states.armies.shape[-1])
        return jax.tree.map(
            lambda value: jnp.broadcast_to(value, (2 * maps, *value.shape)), memory
        )

    initial_candidate_memory = batched_memory(candidate_policy)
    initial_opponent_memory = batched_memory(opponent_policy)

    @jax.jit
    def evaluate(key):
        finished = jnp.zeros((2 * maps,), dtype=jnp.bool_)
        outcomes = jnp.full((2 * maps,), 0.5, dtype=jnp.float32)
        finish_turn = jnp.full((2 * maps,), truncation, dtype=jnp.int32)

        def step(carry, turn):
            (
                states,
                rng,
                candidate_memory,
                opponent_memory,
                finished,
                outcomes,
                finish_turn,
            ) = carry
            observation_zero = jax.vmap(lambda state: get_observation(state, 0))(states)
            observation_one = jax.vmap(lambda state: get_observation(state, 1))(states)
            candidate_observation = jax.tree.map(
                lambda zero, one: jnp.where(
                    candidate_is_zero.reshape(
                        (-1,) + (1,) * (zero.ndim - 1)
                    ),
                    zero,
                    one,
                ),
                observation_zero,
                observation_one,
            )
            opponent_observation = jax.tree.map(
                lambda zero, one: jnp.where(
                    candidate_is_zero.reshape(
                        (-1,) + (1,) * (zero.ndim - 1)
                    ),
                    one,
                    zero,
                ),
                observation_zero,
                observation_one,
            )

            keys = jax.random.split(rng, 4 * maps + 1)
            rng = keys[0]
            candidate_actions, candidate_memory = jax.vmap(candidate_policy.step)(
                keys[1 : 2 * maps + 1],
                candidate_observation,
                states.board_mask,
                candidate_memory,
            )
            opponent_actions, opponent_memory = jax.vmap(opponent_policy.step)(
                keys[2 * maps + 1 :],
                opponent_observation,
                states.board_mask,
                opponent_memory,
            )
            actions_zero = jnp.where(
                candidate_is_zero[:, None], candidate_actions, opponent_actions
            )
            actions_one = jnp.where(
                candidate_is_zero[:, None], opponent_actions, candidate_actions
            )
            timesteps, states = jax.vmap(
                lambda state, actions: environment.step(state, actions, pool)
            )(states, jnp.stack([actions_zero, actions_one], axis=1))

            done = timesteps.terminated | timesteps.truncated
            newly_finished = done & ~finished
            candidate_won = jnp.where(
                candidate_is_zero,
                timesteps.info.winner == 0,
                timesteps.info.winner == 1,
            )
            candidate_lost = jnp.where(
                candidate_is_zero,
                timesteps.info.winner == 1,
                timesteps.info.winner == 0,
            )
            result = jnp.where(
                candidate_won, 1.0, jnp.where(candidate_lost, 0.0, 0.5)
            )
            outcomes = jnp.where(newly_finished, result, outcomes)
            finish_turn = jnp.where(newly_finished, turn + 1, finish_turn)
            finished |= done
            return (
                states,
                rng,
                candidate_memory,
                opponent_memory,
                finished,
                outcomes,
                finish_turn,
            ), None

        (_, _, _, _, finished, outcomes, finish_turn), _ = jax.lax.scan(
            step,
            (
                initial_states,
                key,
                initial_candidate_memory,
                initial_opponent_memory,
                finished,
                outcomes,
                finish_turn,
            ),
            jnp.arange(truncation),
        )
        wins = jnp.sum((outcomes == 1.0) & finished)
        losses = jnp.sum((outcomes == 0.0) & finished)
        draws = 2 * maps - wins - losses
        paired_scores = (outcomes[:maps] + outcomes[maps:]) / 2.0
        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "score": outcomes.mean(),
            "paired_score_std": paired_scores.std(),
            "mean_finish_turn": finish_turn.astype(jnp.float32).mean(),
        }

    return evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="boss")
    parser.add_argument("--maps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--truncation", type=int, default=1200)
    parser.add_argument("--opponents", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.opponents is None:
        args.opponents = [
            opponent for opponent in DEFAULT_OPPONENTS if opponent != args.candidate
        ]

    pool_size = max(16, ((args.maps + 15) // 16) * 16)
    environment = GeneralsEnv(mode="competition", pool_size=pool_size)
    pool_key = jax.random.PRNGKey(args.seed)
    print(f"Generating {pool_size} locked competition maps...")
    pool, _ = environment.reset(pool_key)

    results = {}
    started = time.perf_counter()
    for index, opponent in enumerate(args.opponents):
        print(f"Benchmarking {args.candidate} vs {opponent}...", flush=True)
        evaluator = make_evaluator(
            environment,
            pool,
            args.maps,
            args.truncation,
            args.candidate,
            opponent,
        )
        key = jax.random.fold_in(pool_key, index + 1)
        matchup_started = time.perf_counter()
        raw = evaluator(key)
        raw["score"].block_until_ready()
        result = {name: float(value) for name, value in raw.items()}
        result["games"] = 2 * args.maps
        result["elapsed_seconds"] = time.perf_counter() - matchup_started
        results[opponent] = result
        print(
            f"  {int(result['wins'])}W/{int(result['losses'])}L/"
            f"{int(result['draws'])}D score={result['score']:.3f} "
            f"mean_turn={result['mean_finish_turn']:.1f}"
        )

    games = sum(result["games"] for result in results.values())
    wins = sum(result["wins"] for result in results.values())
    losses = sum(result["losses"] for result in results.values())
    draws = sum(result["draws"] for result in results.values())
    payload = {
        "candidate": args.candidate,
        "seed": args.seed,
        "maps_per_opponent": args.maps,
        "games_per_opponent": 2 * args.maps,
        "truncation": args.truncation,
        "opponents": results,
        "aggregate": {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "games": games,
            "score": (wins + 0.5 * draws) / games,
            "macro_score": sum(r["score"] for r in results.values()) / len(results),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        f"Aggregate: {int(wins)}W/{int(losses)}L/{int(draws)}D "
        f"score={payload['aggregate']['score']:.3f}"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

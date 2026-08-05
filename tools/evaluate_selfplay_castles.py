"""Measure castle construction in stochastic checkpoint self-play.

This uses the same categorical action sampling, observation memory, legal mask,
competition environment, and final curriculum stage as training.  It runs
headlessly in fixed-size batches and emits one aggregate JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from generals.core.game import get_observation
from generals.training.actions import (
    CELL_COUNT,
    MOVE_PLANES,
    PASS_INDEX,
    decode_action,
    legal_action_mask,
)
from generals.training.config import TrainingConfig
from generals.training.observation import augment_observation, init_observation_memory, temporal_input
from generals.training.train import _learning_rate, _load_checkpoint_state, build_network, make_environment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(config: TrainingConfig, checkpoint: Path):
    key = jax.random.PRNGKey(config.seed)
    key, network_key = jax.random.split(key)
    network = build_network(config, network_key)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(partial(_learning_rate, config)),
    )
    optimizer_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    skeleton = (network, optimizer_state, network, jnp.int32(0), jnp.int32(0), key)
    raw, _, ema, iteration, stage_index, _ = _load_checkpoint_state(
        checkpoint, skeleton, config
    )
    return raw, ema, int(iteration), int(stage_index)


def _batched_memory(config: TrainingConfig, games: int):
    memory = init_observation_memory(
        config.pad_to, config.history_size, config.temporal_window
    )
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (games, *value.shape)), memory
    )


@eqx.filter_jit
def evaluate_batch(
    network, pool, key, config: TrainingConfig, environment, sampling: str
):
    """Run one game per pool entry and return per-game construction facts."""
    games = pool.armies.shape[0]
    indices = jnp.arange(games, dtype=jnp.int32)
    states = pool._replace(pool_idx=indices)
    memory_zero = _batched_memory(config, games)
    memory_one = _batched_memory(config, games)
    finished = jnp.zeros((games,), dtype=jnp.bool_)
    had_opportunity = jnp.zeros((games, 2), dtype=jnp.bool_)
    had_build = jnp.zeros((games, 2), dtype=jnp.bool_)
    builds = jnp.zeros((games, 2), dtype=jnp.int32)
    opportunity_steps = jnp.zeros((games, 2), dtype=jnp.int32)
    first_build_turn = jnp.full((games, 2), -1, dtype=jnp.int32)
    result = jnp.full((games,), -2, dtype=jnp.int32)
    game_length = jnp.zeros((games,), dtype=jnp.int32)
    confirmed_new_castles = jnp.zeros((games,), dtype=jnp.int32)
    build_probability_sum = jnp.zeros((games, 2), dtype=jnp.float32)
    best_build_probability_sum = jnp.zeros((games, 2), dtype=jnp.float32)
    best_build_margin_sum = jnp.zeros((games, 2), dtype=jnp.float32)
    best_build_rank_sum = jnp.zeros((games, 2), dtype=jnp.int32)
    max_build_probability = jnp.full((games, 2), -1.0, dtype=jnp.float32)
    max_best_build_margin = jnp.full((games, 2), -jnp.inf, dtype=jnp.float32)
    min_best_build_rank = jnp.full((games, 2), PASS_INDEX + 1, dtype=jnp.int32)
    time_opportunity_steps = jnp.zeros((2, 6), dtype=jnp.int32)
    time_build_probability_sum = jnp.zeros((2, 6), dtype=jnp.float32)
    time_builds = jnp.zeros((2, 6), dtype=jnp.int32)
    legal_site_count_sum = jnp.zeros((games, 2), dtype=jnp.int32)
    affordable_site_army_sum = jnp.zeros((games, 2), dtype=jnp.int32)
    first_build_row = jnp.full((games, 2), -1, dtype=jnp.int32)
    first_build_col = jnp.full((games, 2), -1, dtype=jnp.int32)
    first_build_site_army = jnp.full((games, 2), -1, dtype=jnp.int32)
    first_build_action_probability = jnp.full((games, 2), -1.0, dtype=jnp.float32)
    first_build_total_probability = jnp.full((games, 2), -1.0, dtype=jnp.float32)
    first_build_best_rank = jnp.full((games, 2), -1, dtype=jnp.int32)
    first_build_own_army = jnp.full((games, 2), -1, dtype=jnp.int32)
    first_build_opponent_army = jnp.full((games, 2), -1, dtype=jnp.int32)
    first_build_own_land = jnp.full((games, 2), -1, dtype=jnp.int32)
    first_build_opponent_land = jnp.full((games, 2), -1, dtype=jnp.int32)

    def step(carry, _):
        (
            states,
            rng,
            memory_zero,
            memory_one,
            finished,
            had_opportunity,
            had_build,
            builds,
            opportunity_steps,
            first_build_turn,
            result,
            game_length,
            confirmed_new_castles,
            build_probability_sum,
            best_build_probability_sum,
            best_build_margin_sum,
            best_build_rank_sum,
            max_build_probability,
            max_best_build_margin,
            min_best_build_rank,
            time_opportunity_steps,
            time_build_probability_sum,
            time_builds,
            legal_site_count_sum,
            affordable_site_army_sum,
            first_build_row,
            first_build_col,
            first_build_site_army,
            first_build_action_probability,
            first_build_total_probability,
            first_build_best_rank,
            first_build_own_army,
            first_build_opponent_army,
            first_build_own_land,
            first_build_opponent_land,
        ) = carry
        observation_zero = jax.vmap(lambda state: get_observation(state, 0))(states)
        observation_one = jax.vmap(lambda state: get_observation(state, 1))(states)
        observations = jax.tree.map(
            lambda zero, one: jnp.concatenate([zero, one]),
            observation_zero,
            observation_one,
        )
        memories = jax.tree.map(
            lambda zero, one: jnp.concatenate([zero, one]),
            memory_zero,
            memory_one,
        )
        board_masks = jnp.concatenate([states.board_mask, states.board_mask])
        augmented, memories = jax.vmap(
            lambda observation, memory, board_mask: augment_observation(
                observation,
                memory,
                board_mask,
                config.observation_schema,
                environment.deathtouch_turn or 800,
            )
        )(observations, memories, board_masks)
        histories = temporal_input(memories)
        masks = jax.vmap(legal_action_mask)(observations, board_masks)

        split_keys = jax.random.split(rng, 2 * games + 1)
        logits = jax.vmap(network.forward)(augmented, histories, masks)[0]
        action_indices = (
            jnp.argmax(logits, axis=-1)
            if sampling == "greedy"
            else jax.vmap(jax.random.categorical)(split_keys[1:], logits)
        )
        actions = jax.vmap(decode_action)(action_indices.astype(jnp.int32))
        rng = split_keys[0]
        memory_zero = jax.tree.map(lambda value: value[:games], memories)
        memory_one = jax.tree.map(lambda value: value[games:], memories)

        actions_by_seat = jnp.stack([actions[:games], actions[games:]], axis=1)
        masks_by_seat = jnp.stack([masks[:games], masks[games:]], axis=1)
        active = ~finished
        active_by_seat = active[:, None]
        build = active_by_seat & (actions_by_seat[:, :, 0] == 2)
        build_slice = slice(MOVE_PLANES * CELL_COUNT, PASS_INDEX)
        legal_builds = masks_by_seat[:, :, build_slice]
        opportunity = active_by_seat & jnp.any(legal_builds, axis=-1)
        logits_by_seat = jnp.stack([logits[:games], logits[games:]], axis=1)
        probabilities = jax.nn.softmax(logits_by_seat, axis=-1)
        build_probabilities = probabilities[:, :, build_slice]
        total_build_probability = build_probabilities.sum(axis=-1)
        best_build_probability = jnp.max(
            jnp.where(legal_builds, build_probabilities, 0.0), axis=-1
        )
        best_build_logit = jnp.max(
            logits_by_seat[:, :, build_slice], axis=-1
        )
        nonbuild_logits = jnp.concatenate(
            [
                logits_by_seat[:, :, : MOVE_PLANES * CELL_COUNT],
                logits_by_seat[:, :, PASS_INDEX:],
            ],
            axis=-1,
        )
        best_nonbuild_logit = jnp.max(nonbuild_logits, axis=-1)
        best_build_margin = best_build_logit - best_nonbuild_logit
        best_build_rank = 1 + jnp.sum(
            logits_by_seat > best_build_logit[:, :, None], axis=-1
        )
        legal_site_count = legal_builds.sum(axis=-1)
        armies_by_seat = jnp.stack(
            [observation_zero.armies, observation_one.armies], axis=1
        )
        affordable_site_army = jnp.max(
            jnp.where(
                legal_builds,
                armies_by_seat.reshape(games, 2, -1),
                0,
            ),
            axis=-1,
        )
        action_indices_by_seat = jnp.stack(
            [action_indices[:games], action_indices[games:]], axis=1
        )
        selected_action_probability = jnp.take_along_axis(
            probabilities, action_indices_by_seat[:, :, None], axis=-1
        )[:, :, 0]
        rows = actions_by_seat[:, :, 1]
        cols = actions_by_seat[:, :, 2]
        game_indices = jnp.arange(games)[:, None]
        seat_indices = jnp.arange(2)[None, :]
        build_site_army = armies_by_seat[game_indices, seat_indices, rows, cols]
        own_army = jnp.stack(
            [observation_zero.owned_army_count, observation_one.owned_army_count],
            axis=1,
        )
        opponent_army = jnp.stack(
            [
                observation_zero.opponent_army_count,
                observation_one.opponent_army_count,
            ],
            axis=1,
        )
        own_land = jnp.stack(
            [observation_zero.owned_land_count, observation_one.owned_land_count],
            axis=1,
        )
        opponent_land = jnp.stack(
            [
                observation_zero.opponent_land_count,
                observation_one.opponent_land_count,
            ],
            axis=1,
        )
        turn = states.time[:, None] + 1
        first_build = build & (first_build_turn < 0)
        first_build_turn = jnp.where(
            first_build, turn, first_build_turn
        )
        first_build_row = jnp.where(first_build, rows, first_build_row)
        first_build_col = jnp.where(first_build, cols, first_build_col)
        first_build_site_army = jnp.where(
            first_build, build_site_army, first_build_site_army
        )
        first_build_action_probability = jnp.where(
            first_build, selected_action_probability, first_build_action_probability
        )
        first_build_total_probability = jnp.where(
            first_build, total_build_probability, first_build_total_probability
        )
        first_build_best_rank = jnp.where(
            first_build, best_build_rank, first_build_best_rank
        )
        first_build_own_army = jnp.where(first_build, own_army, first_build_own_army)
        first_build_opponent_army = jnp.where(
            first_build, opponent_army, first_build_opponent_army
        )
        first_build_own_land = jnp.where(first_build, own_land, first_build_own_land)
        first_build_opponent_land = jnp.where(
            first_build, opponent_land, first_build_opponent_land
        )
        had_opportunity |= opportunity
        had_build |= build
        builds += build.astype(jnp.int32)
        opportunity_steps += opportunity.astype(jnp.int32)
        build_probability_sum += jnp.where(
            opportunity, total_build_probability, 0.0
        )
        best_build_probability_sum += jnp.where(
            opportunity, best_build_probability, 0.0
        )
        best_build_margin_sum += jnp.where(opportunity, best_build_margin, 0.0)
        best_build_rank_sum += jnp.where(opportunity, best_build_rank, 0)
        max_build_probability = jnp.where(
            opportunity,
            jnp.maximum(max_build_probability, total_build_probability),
            max_build_probability,
        )
        max_best_build_margin = jnp.where(
            opportunity,
            jnp.maximum(max_best_build_margin, best_build_margin),
            max_best_build_margin,
        )
        min_best_build_rank = jnp.where(
            opportunity,
            jnp.minimum(min_best_build_rank, best_build_rank),
            min_best_build_rank,
        )
        legal_site_count_sum += jnp.where(opportunity, legal_site_count, 0)
        affordable_site_army_sum += jnp.where(opportunity, affordable_site_army, 0)
        time_bin = jnp.minimum(states.time // 200, 5)
        time_one_hot = jax.nn.one_hot(time_bin, 6, dtype=jnp.int32)
        opportunity_time = opportunity[:, :, None] & time_one_hot[:, None, :].astype(bool)
        time_opportunity_steps += opportunity_time.sum(axis=0)
        time_build_probability_sum += jnp.where(
            opportunity_time,
            total_build_probability[:, :, None],
            0.0,
        ).sum(axis=0)
        time_builds += jnp.where(
            opportunity_time, build[:, :, None], False
        ).sum(axis=0)

        timesteps, next_states = jax.vmap(
            lambda state, action: environment.step(state, action, pool)
        )(states, actions_by_seat)
        born = timesteps.last_state.castles & ~states.castles
        confirmed_new_castles += (
            born.reshape(games, -1).sum(axis=1) * active.astype(jnp.int32)
        )
        done = timesteps.terminated | timesteps.truncated
        newly_finished = done & ~finished
        result = jnp.where(newly_finished, timesteps.info.winner, result)
        game_length = jnp.where(newly_finished, timesteps.info.time, game_length)
        finished |= done

        return (
            next_states,
            rng,
            memory_zero,
            memory_one,
            finished,
            had_opportunity,
            had_build,
            builds,
            opportunity_steps,
            first_build_turn,
            result,
            game_length,
            confirmed_new_castles,
            build_probability_sum,
            best_build_probability_sum,
            best_build_margin_sum,
            best_build_rank_sum,
            max_build_probability,
            max_best_build_margin,
            min_best_build_rank,
            time_opportunity_steps,
            time_build_probability_sum,
            time_builds,
            legal_site_count_sum,
            affordable_site_army_sum,
            first_build_row,
            first_build_col,
            first_build_site_army,
            first_build_action_probability,
            first_build_total_probability,
            first_build_best_rank,
            first_build_own_army,
            first_build_opponent_army,
            first_build_own_land,
            first_build_opponent_land,
        ), None

    final, _ = jax.lax.scan(
        step,
        (
            states,
            key,
            memory_zero,
            memory_one,
            finished,
            had_opportunity,
            had_build,
            builds,
            opportunity_steps,
            first_build_turn,
            result,
            game_length,
            confirmed_new_castles,
            build_probability_sum,
            best_build_probability_sum,
            best_build_margin_sum,
            best_build_rank_sum,
            max_build_probability,
            max_best_build_margin,
            min_best_build_rank,
            time_opportunity_steps,
            time_build_probability_sum,
            time_builds,
            legal_site_count_sum,
            affordable_site_army_sum,
            first_build_row,
            first_build_col,
            first_build_site_army,
            first_build_action_probability,
            first_build_total_probability,
            first_build_best_rank,
            first_build_own_army,
            first_build_opponent_army,
            first_build_own_land,
            first_build_opponent_land,
        ),
        None,
        length=config.truncation,
    )
    return {
        "finished": final[4],
        "had_opportunity": final[5],
        "had_build": final[6],
        "builds": final[7],
        "opportunity_steps": final[8],
        "first_build_turn": final[9],
        "winner": final[10],
        "game_length": final[11],
        "confirmed_new_castles": final[12],
        "build_probability_sum": final[13],
        "best_build_probability_sum": final[14],
        "best_build_margin_sum": final[15],
        "best_build_rank_sum": final[16],
        "max_build_probability": final[17],
        "max_best_build_margin": final[18],
        "min_best_build_rank": final[19],
        "time_opportunity_steps": final[20],
        "time_build_probability_sum": final[21],
        "time_builds": final[22],
        "legal_site_count_sum": final[23],
        "affordable_site_army_sum": final[24],
        "first_build_row": final[25],
        "first_build_col": final[26],
        "first_build_site_army": final[27],
        "first_build_action_probability": final[28],
        "first_build_total_probability": final[29],
        "first_build_best_rank": final[30],
        "first_build_own_army": final[31],
        "first_build_opponent_army": final[32],
        "first_build_own_land": final[33],
        "first_build_opponent_land": final[34],
    }


def _wilson(successes: int, trials: int) -> list[float]:
    if trials == 0:
        return [math.nan, math.nan]
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / denominator
    return [center - radius, center + radius]


@dataclass
class Aggregate:
    had_opportunity: list[np.ndarray] = field(default_factory=list)
    had_build: list[np.ndarray] = field(default_factory=list)
    builds: list[np.ndarray] = field(default_factory=list)
    opportunity_steps: list[np.ndarray] = field(default_factory=list)
    first_build_turn: list[np.ndarray] = field(default_factory=list)
    winner: list[np.ndarray] = field(default_factory=list)
    game_length: list[np.ndarray] = field(default_factory=list)
    confirmed_new_castles: list[np.ndarray] = field(default_factory=list)
    finished: list[np.ndarray] = field(default_factory=list)
    build_probability_sum: list[np.ndarray] = field(default_factory=list)
    best_build_probability_sum: list[np.ndarray] = field(default_factory=list)
    best_build_margin_sum: list[np.ndarray] = field(default_factory=list)
    best_build_rank_sum: list[np.ndarray] = field(default_factory=list)
    max_build_probability: list[np.ndarray] = field(default_factory=list)
    max_best_build_margin: list[np.ndarray] = field(default_factory=list)
    min_best_build_rank: list[np.ndarray] = field(default_factory=list)
    time_opportunity_steps: list[np.ndarray] = field(default_factory=list)
    time_build_probability_sum: list[np.ndarray] = field(default_factory=list)
    time_builds: list[np.ndarray] = field(default_factory=list)
    legal_site_count_sum: list[np.ndarray] = field(default_factory=list)
    affordable_site_army_sum: list[np.ndarray] = field(default_factory=list)
    first_build_row: list[np.ndarray] = field(default_factory=list)
    first_build_col: list[np.ndarray] = field(default_factory=list)
    first_build_site_army: list[np.ndarray] = field(default_factory=list)
    first_build_action_probability: list[np.ndarray] = field(default_factory=list)
    first_build_total_probability: list[np.ndarray] = field(default_factory=list)
    first_build_best_rank: list[np.ndarray] = field(default_factory=list)
    first_build_own_army: list[np.ndarray] = field(default_factory=list)
    first_build_opponent_army: list[np.ndarray] = field(default_factory=list)
    first_build_own_land: list[np.ndarray] = field(default_factory=list)
    first_build_opponent_land: list[np.ndarray] = field(default_factory=list)

    def add(self, batch) -> None:
        host = jax.device_get(batch)
        for name in self.__dataclass_fields__:
            getattr(self, name).append(np.asarray(host[name]))

    def summarize(self) -> dict:
        additive = {"time_opportunity_steps", "time_build_probability_sum", "time_builds"}
        values = {}
        for name in self.__dataclass_fields__:
            batches = getattr(self, name)
            values[name] = (
                np.stack(batches).sum(axis=0)
                if name in additive
                else np.concatenate(batches, axis=0)
            )
        had_build = values["had_build"]
        had_opportunity = values["had_opportunity"]
        builds = values["builds"]
        either = had_build.any(axis=1)
        both = had_build.all(axis=1)
        either_opportunity = had_opportunity.any(axis=1)
        games = len(either)
        player_games = 2 * games
        player_games_with_build = int(had_build.sum())
        player_games_with_opportunity = int(had_opportunity.sum())
        opportunity_player_games_that_built = int((had_build & had_opportunity).sum())
        first_turns = values["first_build_turn"][values["first_build_turn"] >= 0]
        game_lengths = values["game_length"]
        truncated = game_lengths >= 1200
        decisive_lengths = game_lengths[~truncated]
        build_games_by_winner = {
            "player_0": int((either & (values["winner"] == 0)).sum()),
            "player_1": int((either & (values["winner"] == 1)).sum()),
            "draw": int((either & (values["winner"] < 0)).sum()),
        }
        opportunity_steps = int(values["opportunity_steps"].sum())
        opportunity_mask = values["had_opportunity"]
        max_probabilities = values["max_build_probability"][opportunity_mask]
        max_margins = values["max_best_build_margin"][opportunity_mask]
        min_ranks = values["min_best_build_rank"][opportunity_mask]
        time_rows = []
        for index, start in enumerate(range(0, 1200, 200)):
            steps = int(values["time_opportunity_steps"][:, index].sum())
            probability_sum = float(
                values["time_build_probability_sum"][:, index].sum()
            )
            time_rows.append(
                {
                    "turns": f"{start}-{start + 199}",
                    "legal_opportunity_player_steps": steps,
                    "actual_builds": int(values["time_builds"][:, index].sum()),
                    "mean_aggregate_build_probability": (
                        probability_sum / steps if steps else None
                    ),
                }
            )
        build_events = []
        for game_index, seat in np.argwhere(had_build):
            winner = int(values["winner"][game_index])
            build_events.append(
                {
                    "game_index": int(game_index),
                    "builder_seat": int(seat),
                    "turn": int(values["first_build_turn"][game_index, seat]),
                    "row": int(values["first_build_row"][game_index, seat]),
                    "col": int(values["first_build_col"][game_index, seat]),
                    "site_army_before_build": int(
                        values["first_build_site_army"][game_index, seat]
                    ),
                    "selected_build_probability": float(
                        values["first_build_action_probability"][game_index, seat]
                    ),
                    "aggregate_build_probability": float(
                        values["first_build_total_probability"][game_index, seat]
                    ),
                    "best_build_rank": int(
                        values["first_build_best_rank"][game_index, seat]
                    ),
                    "own_army": int(values["first_build_own_army"][game_index, seat]),
                    "opponent_army": int(
                        values["first_build_opponent_army"][game_index, seat]
                    ),
                    "own_land": int(values["first_build_own_land"][game_index, seat]),
                    "opponent_land": int(
                        values["first_build_opponent_land"][game_index, seat]
                    ),
                    "winner": winner,
                    "builder_won": winner == int(seat),
                    "game_length": int(values["game_length"][game_index]),
                }
            )
        return {
            "games": games,
            "finished_games": int(values["finished"].sum()),
            "games_with_build_either_side": int(either.sum()),
            "games_with_build_either_side_rate": float(either.mean()),
            "games_with_build_either_side_wilson_95": _wilson(int(either.sum()), games),
            "games_with_build_both_sides": int(both.sum()),
            "games_with_build_exactly_one_side": int((either & ~both).sum()),
            "games_with_no_build": int((~either).sum()),
            "games_with_legal_build_opportunity_either_side": int(either_opportunity.sum()),
            "seat_0_games_with_build": int(had_build[:, 0].sum()),
            "seat_1_games_with_build": int(had_build[:, 1].sum()),
            "player_games": player_games,
            "player_games_with_build": player_games_with_build,
            "player_game_build_rate": player_games_with_build / player_games,
            "player_games_with_legal_build_opportunity": player_games_with_opportunity,
            "opportunity_player_games_that_built": opportunity_player_games_that_built,
            "build_given_player_game_opportunity_rate": (
                opportunity_player_games_that_built / player_games_with_opportunity
                if player_games_with_opportunity
                else math.nan
            ),
            "total_build_actions": int(builds.sum()),
            "seat_0_build_actions": int(builds[:, 0].sum()),
            "seat_1_build_actions": int(builds[:, 1].sum()),
            "total_confirmed_new_castles": int(values["confirmed_new_castles"].sum()),
            "total_legal_build_opportunity_player_steps": opportunity_steps,
            "build_per_legal_opportunity_step_rate": (
                float(builds.sum() / opportunity_steps)
                if opportunity_steps
                else math.nan
            ),
            "logit_diagnostics_on_legal_opportunity_steps": {
                "expected_build_actions_from_probability_mass": float(
                    values["build_probability_sum"].sum()
                ),
                "mean_aggregate_build_probability": (
                    float(values["build_probability_sum"].sum() / opportunity_steps)
                    if opportunity_steps
                    else None
                ),
                "mean_best_individual_build_probability": (
                    float(values["best_build_probability_sum"].sum() / opportunity_steps)
                    if opportunity_steps
                    else None
                ),
                "mean_best_build_logit_margin_vs_best_nonbuild": (
                    float(values["best_build_margin_sum"].sum() / opportunity_steps)
                    if opportunity_steps
                    else None
                ),
                "mean_best_build_rank": (
                    float(values["best_build_rank_sum"].sum() / opportunity_steps)
                    if opportunity_steps
                    else None
                ),
                "per_player_game_max_aggregate_build_probability": {
                    "median": float(np.median(max_probabilities)),
                    "p90": float(np.quantile(max_probabilities, 0.90)),
                    "p99": float(np.quantile(max_probabilities, 0.99)),
                    "max": float(np.max(max_probabilities)),
                },
                "per_player_game_best_logit_margin": {
                    "median": float(np.median(max_margins)),
                    "p90": float(np.quantile(max_margins, 0.90)),
                    "p99": float(np.quantile(max_margins, 0.99)),
                    "max": float(np.max(max_margins)),
                },
                "per_player_game_best_rank": {
                    "median": float(np.median(min_ranks)),
                    "p10": float(np.quantile(min_ranks, 0.10)),
                    "p1": float(np.quantile(min_ranks, 0.01)),
                    "best": int(np.min(min_ranks)),
                },
                "mean_legal_build_sites": (
                    float(values["legal_site_count_sum"].sum() / opportunity_steps)
                    if opportunity_steps
                    else None
                ),
                "mean_army_on_strongest_affordable_site": (
                    float(values["affordable_site_army_sum"].sum() / opportunity_steps)
                    if opportunity_steps
                    else None
                ),
                "by_turn": time_rows,
            },
            "first_build_turn": {
                "count": int(len(first_turns)),
                "median": float(np.median(first_turns)) if len(first_turns) else None,
                "p10": float(np.quantile(first_turns, 0.10)) if len(first_turns) else None,
                "p90": float(np.quantile(first_turns, 0.90)) if len(first_turns) else None,
            },
            "game_length": {
                "mean": float(game_lengths.mean()),
                "median": float(np.median(game_lengths)),
                "p10": float(np.quantile(game_lengths, 0.10)),
                "p25": float(np.quantile(game_lengths, 0.25)),
                "p75": float(np.quantile(game_lengths, 0.75)),
                "p90": float(np.quantile(game_lengths, 0.90)),
                "truncated_at_1200": int(truncated.sum()),
                "truncation_rate": float(truncated.mean()),
                "decisive_games": int(len(decisive_lengths)),
                "decisive_mean": (
                    float(decisive_lengths.mean())
                    if len(decisive_lengths)
                    else None
                ),
                "decisive_median": (
                    float(np.median(decisive_lengths))
                    if len(decisive_lengths)
                    else None
                ),
            },
            "mean_game_length": float(game_lengths.mean()),
            "games_with_build_by_result": build_games_by_winner,
            "first_build_events": build_events,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--games", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--policy", choices=("raw", "ema", "both"), default="both")
    parser.add_argument(
        "--sampling", choices=("categorical", "greedy"), default="categorical"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.games < 1 or args.batch_size < 1:
        parser.error("--games and --batch-size must be positive")
    if args.games % args.batch_size:
        parser.error("--games must be divisible by --batch-size")
    if args.batch_size % 16:
        parser.error("--batch-size must be divisible by 16 for all 18-21 map shapes")
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = TrainingConfig.from_toml(args.config)
    raw, ema, iteration, stage_index = load_checkpoint(config, args.checkpoint)
    if stage_index != len(config.curriculum) - 1:
        raise ValueError(
            f"Checkpoint is at curriculum stage {stage_index}; expected final stage "
            f"{len(config.curriculum) - 1}"
        )
    policies = {"raw": raw, "ema": ema}
    selected_names = tuple(policies) if args.policy == "both" else (args.policy,)
    aggregates = {name: Aggregate() for name in selected_names}
    environment = make_environment(
        config, config.curriculum[stage_index], pool_size=args.batch_size
    )

    for batch_index in range(args.games // args.batch_size):
        batch_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), batch_index)
        pool_key, policy_key = jax.random.split(batch_key)
        pool, _ = environment.reset(pool_key)
        for name in selected_names:
            result = evaluate_batch(
                policies[name], pool, policy_key, config, environment, args.sampling
            )
            aggregates[name].add(result)
        print(
            f"completed batch {batch_index + 1}/{args.games // args.batch_size} "
            f"({(batch_index + 1) * args.batch_size} games per policy)",
            flush=True,
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_iteration": iteration,
        "checkpoint_stage": stage_index,
        "config": str(args.config),
        "sampling": (
            "categorical (same logits and sampling rule as training rollout)"
            if args.sampling == "categorical"
            else "greedy argmax"
        ),
        "map_distribution": "final curriculum stage, exact competition generator",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "elapsed_seconds": time.perf_counter() - started,
        "policies": {
            name: aggregates[name].summarize() for name in selected_names
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

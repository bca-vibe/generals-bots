"""Build a paired counterfactual atlas of legal castle opportunities.

The source states come from stochastic self-play under the checkpoint policy.
Within each 200-turn stratum, reservoir sampling keeps at most one legal-build
state per game and player seat.  At every selected state, matched continuations
compare two interventions:

* control: force the model's highest-logit legal non-build action;
* build: force the model's highest-logit legal castle site.

All later actions are sampled from the checkpoint policy.  Each branch receives
the same categorical random keys (common random numbers), including the other
player's action on the intervention turn.  Outcomes are therefore paired by
state, repetition, and future policy noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
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
    build_cost_grid_from_observation,
    decode_action,
    legal_action_mask,
)
from generals.training.config import TrainingConfig
from generals.training.observation import (
    augment_observation,
    init_observation_memory,
    temporal_input,
)
from generals.training.train import (
    _learning_rate,
    _load_checkpoint_state,
    build_network,
    make_environment,
)

BUILD_START = MOVE_PLANES * CELL_COUNT
TURN_BIN_WIDTH = 200
TURN_BINS = 6
BRANCH_CONTROL = 0
BRANCH_BUILD = 1
HORIZONS = (25, 50, 100, 200)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(*trees) -> str:
    """Hash selected state/memory tensors so A/B common-state use is auditable."""
    digest = hashlib.sha256()
    for tree in trees:
        for leaf in jax.tree.leaves(tree):
            array = np.asarray(jax.device_get(leaf))
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes(order="C"))
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
    raw, _, ema, iteration, stage_index, _ = _load_checkpoint_state(checkpoint, skeleton, config)
    return raw, ema, int(iteration), int(stage_index)


def _batched_memory(config: TrainingConfig, games: int):
    memory = init_observation_memory(config.pad_to, config.history_size, config.temporal_window)
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (games, *value.shape)), memory)


def _select_by_seat(zero, one, seats):
    return jax.tree.map(
        lambda z, o: jnp.where(seats.reshape(seats.shape + (1,) * (z.ndim - seats.ndim)), z, o),
        one,
        zero,
    )


def _slot_broadcast(tree, games: int):
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value[:, None, None], (games, 2, TURN_BINS, *value.shape[1:])),
        tree,
    )


def _replace_slots(selected, source, replace):
    def replace_leaf(old, new):
        expanded = new[:, None, None]
        condition = replace.reshape(replace.shape + (1,) * (expanded.ndim - replace.ndim))
        return jnp.where(condition, expanded, old)

    return jax.tree.map(replace_leaf, selected, source)


@eqx.filter_jit
def collect_opportunity_batch(network, pool, key, config, environment):
    """Reservoir-sample one opportunity per game, seat, and turn bin."""
    games = pool.armies.shape[0]
    states = pool._replace(pool_idx=jnp.arange(games, dtype=jnp.int32))
    memory_zero = _batched_memory(config, games)
    memory_one = _batched_memory(config, games)
    sampled_states = _slot_broadcast(states, games)
    sampled_memory_zero = _slot_broadcast(memory_zero, games)
    sampled_memory_one = _slot_broadcast(memory_one, games)
    seen = jnp.zeros((games, 2, TURN_BINS), dtype=jnp.int32)
    found = jnp.zeros((games, 2, TURN_BINS), dtype=jnp.bool_)
    finished = jnp.zeros((games,), dtype=jnp.bool_)

    def step(carry, _):
        (
            states,
            rng,
            memory_zero,
            memory_one,
            sampled_states,
            sampled_memory_zero,
            sampled_memory_one,
            seen,
            found,
            finished,
        ) = carry
        prior_memory_zero = memory_zero
        prior_memory_one = memory_one
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
        masks = jax.vmap(legal_action_mask)(observations, board_masks)
        histories = temporal_input(memories)
        logits = jax.vmap(network.forward)(augmented, histories, masks)[0]

        masks_by_seat = jnp.stack([masks[:games], masks[games:]], axis=1)
        opportunity = (~finished[:, None]) & jnp.any(masks_by_seat[:, :, BUILD_START:PASS_INDEX], axis=-1)
        turn_bin = jnp.minimum(states.time // TURN_BIN_WIDTH, TURN_BINS - 1)
        in_bin = jax.nn.one_hot(turn_bin, TURN_BINS, dtype=jnp.bool_)
        opportunity_by_bin = opportunity[:, :, None] & in_bin[:, None, :]
        next_seen = seen + opportunity_by_bin.astype(jnp.int32)

        rng, reservoir_key, action_key = jax.random.split(rng, 3)
        reservoir_draw = jax.random.uniform(reservoir_key, (games, 2, 1))
        replace = opportunity_by_bin & (reservoir_draw < 1.0 / jnp.maximum(next_seen, 1))
        sampled_states = _replace_slots(sampled_states, states, replace)
        sampled_memory_zero = _replace_slots(sampled_memory_zero, prior_memory_zero, replace)
        sampled_memory_one = _replace_slots(sampled_memory_one, prior_memory_one, replace)
        seen = next_seen
        found |= opportunity_by_bin

        action_keys = jax.random.split(action_key, 2 * games)
        action_indices = jax.vmap(jax.random.categorical)(action_keys, logits)
        actions = jax.vmap(decode_action)(action_indices.astype(jnp.int32))
        actions_by_seat = jnp.stack([actions[:games], actions[games:]], axis=1)
        timesteps, _ = jax.vmap(lambda state, action: environment.step(state, action, pool))(states, actions_by_seat)
        active = ~finished
        states = jax.tree.map(
            lambda old, new: jnp.where(active.reshape(active.shape + (1,) * (old.ndim - 1)), new, old),
            states,
            timesteps.last_state,
        )
        memory_zero = jax.tree.map(lambda value: value[:games], memories)
        memory_one = jax.tree.map(lambda value: value[games:], memories)
        finished |= timesteps.terminated | timesteps.truncated
        return (
            states,
            rng,
            memory_zero,
            memory_one,
            sampled_states,
            sampled_memory_zero,
            sampled_memory_one,
            seen,
            found,
            finished,
        ), None

    initial = (
        states,
        key,
        memory_zero,
        memory_one,
        sampled_states,
        sampled_memory_zero,
        sampled_memory_one,
        seen,
        found,
        finished,
    )
    final, _ = jax.lax.scan(step, initial, None, length=config.truncation)
    return {
        "states": final[4],
        "memory_zero": final[5],
        "memory_one": final[6],
        "opportunity_steps": final[7],
        "found": final[8],
    }


def _expand_pair_axis(value, repetitions: int):
    states = value.shape[0]
    return jnp.broadcast_to(value[:, None, None], (states, repetitions, 2, *value.shape[1:])).reshape(
        states * repetitions * 2, *value.shape[1:]
    )


def _minimum_manhattan(mask, row, col):
    rows = jnp.arange(mask.shape[-2])[:, None]
    cols = jnp.arange(mask.shape[-1])[None, :]
    distances = jnp.abs(rows - row) + jnp.abs(cols - col)
    return jnp.min(jnp.where(mask, distances, 999))


@eqx.filter_jit
def evaluate_pair_batch(
    network,
    critic_networks,
    source_states,
    source_memory_zero,
    source_memory_one,
    actor_seats,
    pool,
    key,
    config,
    environment,
    *,
    repetitions: int,
    rollout_steps: int,
):
    """Evaluate paired forced-build and forced-nonbuild continuations."""
    count = source_states.armies.shape[0]
    observations_zero = jax.vmap(lambda state: get_observation(state, 0))(source_states)
    observations_one = jax.vmap(lambda state: get_observation(state, 1))(source_states)
    observations = jax.tree.map(
        lambda zero, one: jnp.concatenate([zero, one]),
        observations_zero,
        observations_one,
    )
    memories = jax.tree.map(
        lambda zero, one: jnp.concatenate([zero, one]),
        source_memory_zero,
        source_memory_one,
    )
    board_masks = jnp.concatenate([source_states.board_mask, source_states.board_mask])
    augmented, updated_memories = jax.vmap(
        lambda observation, memory, board_mask: augment_observation(
            observation,
            memory,
            board_mask,
            config.observation_schema,
            environment.deathtouch_turn or 800,
        )
    )(observations, memories, board_masks)
    masks = jax.vmap(legal_action_mask)(observations, board_masks)
    histories = temporal_input(updated_memories)
    logits, values, _ = jax.vmap(network.forward)(augmented, histories, masks)
    logits_zero, logits_one = logits[:count], logits[count:]
    masks_zero, masks_one = masks[:count], masks[count:]
    actor_logits = jnp.where(actor_seats[:, None] == 0, logits_zero, logits_one)
    opponent_logits = jnp.where(actor_seats[:, None] == 0, logits_one, logits_zero)
    actor_masks = jnp.where(actor_seats[:, None] == 0, masks_zero, masks_one)
    actor_values = jnp.where(actor_seats == 0, values[:count], values[count:])
    critic_values = (
        jnp.stack(
            [jax.vmap(critic.forward)(augmented, histories, masks)[1] for critic in critic_networks],
            axis=-1,
        )
        if critic_networks
        else jnp.empty((2 * count, 0), dtype=values.dtype)
    )
    critic_actor_values = jnp.where(
        actor_seats[:, None] == 0,
        critic_values[:count],
        critic_values[count:],
    )

    build_offsets = jnp.argmax(actor_logits[:, BUILD_START:PASS_INDEX], axis=-1)
    build_indices = BUILD_START + build_offsets
    control_logits = actor_logits.at[:, BUILD_START:PASS_INDEX].set(-1e9)
    control_indices = jnp.argmax(control_logits, axis=-1)
    probabilities = jax.nn.softmax(actor_logits, axis=-1)
    build_probabilities = probabilities[:, BUILD_START:PASS_INDEX]
    total_build_probability = jnp.sum(build_probabilities, axis=-1)
    best_build_probability = jnp.take_along_axis(probabilities, build_indices[:, None], axis=-1)[:, 0]
    control_probability = jnp.take_along_axis(probabilities, control_indices[:, None], axis=-1)[:, 0]
    best_build_logit = jnp.take_along_axis(actor_logits, build_indices[:, None], axis=-1)[:, 0]
    best_control_logit = jnp.take_along_axis(actor_logits, control_indices[:, None], axis=-1)[:, 0]
    best_build_rank = 1 + jnp.sum(actor_logits > best_build_logit[:, None], axis=-1)
    legal_build_sites = jnp.sum(actor_masks[:, BUILD_START:PASS_INDEX], axis=-1)

    build_actions = jax.vmap(decode_action)(build_indices.astype(jnp.int32))
    control_actions = jax.vmap(decode_action)(control_indices.astype(jnp.int32))
    site_rows = build_actions[:, 1]
    site_cols = build_actions[:, 2]
    actor_observations = _select_by_seat(observations_zero, observations_one, actor_seats)
    site_cost_grids = jax.vmap(build_cost_grid_from_observation)(actor_observations)
    indices = jnp.arange(count)
    site_cost = site_cost_grids[indices, site_rows, site_cols]
    site_army = actor_observations.armies[indices, site_rows, site_cols]
    own_army = actor_observations.owned_army_count
    opponent_army = actor_observations.opponent_army_count
    own_land = actor_observations.owned_land_count
    opponent_land = actor_observations.opponent_land_count
    own_structures = (actor_observations.generals | actor_observations.castles) & actor_observations.owned_cells
    own_generals = actor_observations.generals & actor_observations.owned_cells
    distance_to_general = jax.vmap(_minimum_manhattan)(own_generals, site_rows, site_cols)
    distance_to_structure = jax.vmap(_minimum_manhattan)(own_structures, site_rows, site_cols)
    true_enemy = jnp.where(
        actor_seats[:, None, None] == 0,
        source_states.ownership[:, 1],
        source_states.ownership[:, 0],
    )
    distance_to_enemy_land = jax.vmap(_minimum_manhattan)(true_enemy, site_rows, site_cols)
    own_castles = jnp.sum(actor_observations.castles & actor_observations.owned_cells, axis=(-2, -1))

    features = {
        "turn": source_states.time,
        "actor_seat": actor_seats,
        "site_row": site_rows,
        "site_col": site_cols,
        "site_cost": site_cost,
        "site_army": site_army,
        "post_build_garrison": site_army - site_cost,
        "own_army": own_army,
        "opponent_army": opponent_army,
        "army_margin": own_army - opponent_army,
        "own_land": own_land,
        "opponent_land": opponent_land,
        "land_margin": own_land - opponent_land,
        "own_castles": own_castles,
        "distance_to_general": distance_to_general,
        "distance_to_nearest_own_structure": distance_to_structure,
        "distance_to_enemy_land_true": distance_to_enemy_land,
        "legal_build_sites": legal_build_sites,
        "actor_value": actor_values,
        "common_critic_actor_values": critic_actor_values,
        "total_build_probability": total_build_probability,
        "best_build_probability": best_build_probability,
        "control_probability": control_probability,
        "best_build_logit": best_build_logit,
        "best_control_logit": best_control_logit,
        "best_build_logit_margin": best_build_logit - best_control_logit,
        "best_build_rank": best_build_rank,
        "build_action_index": build_indices,
        "control_action_index": control_indices,
        "control_kind": control_actions[:, 0],
    }

    flat_states = jax.tree.map(lambda value: _expand_pair_axis(value, repetitions), source_states)
    memory_zero = jax.tree.map(
        lambda value: _expand_pair_axis(value, repetitions),
        jax.tree.map(lambda value: value[:count], updated_memories),
    )
    memory_one = jax.tree.map(
        lambda value: _expand_pair_axis(value, repetitions),
        jax.tree.map(lambda value: value[count:], updated_memories),
    )
    actor_flat = _expand_pair_axis(actor_seats, repetitions)
    build_flat = _expand_pair_axis(build_indices, repetitions)
    control_flat = _expand_pair_axis(control_indices, repetitions)
    rows_flat = _expand_pair_axis(site_rows, repetitions)
    cols_flat = _expand_pair_axis(site_cols, repetitions)
    cost_flat = _expand_pair_axis(site_cost, repetitions)
    branch = jnp.broadcast_to(jnp.arange(2)[None, None, :], (count, repetitions, 2)).reshape(-1)
    actor_action_index = jnp.where(branch == BRANCH_BUILD, build_flat, control_flat)

    key, intervention_key = jax.random.split(key)
    opponent_keys = jax.random.split(intervention_key, count * repetitions)
    repeated_opponent_logits = jnp.broadcast_to(
        opponent_logits[:, None],
        (count, repetitions, opponent_logits.shape[-1]),
    ).reshape(count * repetitions, -1)
    opponent_index = jax.vmap(jax.random.categorical)(opponent_keys, repeated_opponent_logits)
    opponent_index = jnp.repeat(opponent_index, 2)
    actor_actions = jax.vmap(decode_action)(actor_action_index.astype(jnp.int32))
    opponent_actions = jax.vmap(decode_action)(opponent_index.astype(jnp.int32))
    action_zero = jnp.where(actor_flat[:, None] == 0, actor_actions, opponent_actions)
    action_one = jnp.where(actor_flat[:, None] == 1, actor_actions, opponent_actions)
    actions = jnp.stack([action_zero, action_one], axis=1)
    intervention_timestep, _ = jax.vmap(lambda state, action: environment.step(state, action, pool))(
        flat_states, actions
    )
    states = intervention_timestep.last_state
    done = intervention_timestep.terminated | intervention_timestep.truncated

    # Probe V on the paired successor states before sampling either policy's
    # next action.  This is the value head's direct assessment of the forced
    # action, with the opponent action and recurrent context held fixed across
    # build/control branches.
    post_observations_zero = jax.vmap(lambda state: get_observation(state, 0))(states)
    post_observations_one = jax.vmap(lambda state: get_observation(state, 1))(states)
    post_observations = jax.tree.map(
        lambda zero, one: jnp.concatenate([zero, one]),
        post_observations_zero,
        post_observations_one,
    )
    post_memories = jax.tree.map(lambda zero, one: jnp.concatenate([zero, one]), memory_zero, memory_one)
    post_board_masks = jnp.concatenate([states.board_mask, states.board_mask])
    post_augmented, post_memories = jax.vmap(
        lambda observation, memory, board_mask: augment_observation(
            observation,
            memory,
            board_mask,
            config.observation_schema,
            environment.deathtouch_turn or 800,
        )
    )(post_observations, post_memories, post_board_masks)
    post_masks = jax.vmap(legal_action_mask)(post_observations, post_board_masks)
    post_histories = temporal_input(post_memories)
    _, post_values, _ = jax.vmap(network.forward)(post_augmented, post_histories, post_masks)
    common_post_values = (
        jnp.stack(
            [jax.vmap(critic.forward)(post_augmented, post_histories, post_masks)[1] for critic in critic_networks],
            axis=-1,
        )
        if critic_networks
        else jnp.empty((post_augmented.shape[0], 0), dtype=post_values.dtype)
    )
    games = states.armies.shape[0]
    post_actor_values = jnp.where(actor_flat == 0, post_values[:games], post_values[games:])
    common_post_actor_values = jnp.where(
        actor_flat[:, None] == 0,
        common_post_values[:games],
        common_post_values[games:],
    )
    winner = intervention_timestep.info.winner
    actor_won = winner == actor_flat
    actor_lost = (winner >= 0) & ~actor_won
    outcome = jnp.where(actor_won, 1.0, jnp.where(actor_lost, 0.0, 0.5))
    outcome = jnp.where(done, outcome, 0.5)
    finish_relative_turn = jnp.where(done, 1, -1)
    flat_index = jnp.arange(states.armies.shape[0])
    owns_site = states.ownership[flat_index, actor_flat, rows_flat, cols_flat]
    site_is_castle = states.castles[flat_index, rows_flat, cols_flat]
    ever_lost_site = (branch == BRANCH_BUILD) & ~owns_site
    production_tick = (branch == BRANCH_BUILD) & owns_site & site_is_castle & (states.time % 2 == 0) & (winner < 0)
    production_ticks = production_tick.astype(jnp.int32)
    payback_relative_turn = jnp.where(production_ticks >= cost_flat, 1, -1)
    survived_horizons = jnp.zeros((states.armies.shape[0], len(HORIZONS)), dtype=jnp.bool_)

    def step(carry, scan_index):
        (
            states,
            rng,
            memory_zero,
            memory_one,
            done,
            outcome,
            finish_relative_turn,
            ever_lost_site,
            production_ticks,
            payback_relative_turn,
            survived_horizons,
        ) = carry
        games = states.armies.shape[0]
        observations_zero = jax.vmap(lambda state: get_observation(state, 0))(states)
        observations_one = jax.vmap(lambda state: get_observation(state, 1))(states)
        observations = jax.tree.map(
            lambda zero, one: jnp.concatenate([zero, one]),
            observations_zero,
            observations_one,
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
        masks = jax.vmap(legal_action_mask)(observations, board_masks)
        histories = temporal_input(memories)
        logits = jax.vmap(network.forward)(augmented, histories, masks)[0]

        rng, action_key = jax.random.split(rng)
        paired_games = count * repetitions
        common_keys = jax.random.split(action_key, paired_games * 2).reshape(paired_games, 2, 2)
        keys_zero = jnp.repeat(common_keys[:, 0], 2, axis=0)
        keys_one = jnp.repeat(common_keys[:, 1], 2, axis=0)
        action_zero_index = jax.vmap(jax.random.categorical)(keys_zero, logits[:games])
        action_one_index = jax.vmap(jax.random.categorical)(keys_one, logits[games:])
        actions = jnp.stack(
            [
                jax.vmap(decode_action)(action_zero_index.astype(jnp.int32)),
                jax.vmap(decode_action)(action_one_index.astype(jnp.int32)),
            ],
            axis=1,
        )
        active = ~done
        timesteps, _ = jax.vmap(lambda state, action: environment.step(state, action, pool))(states, actions)
        states = jax.tree.map(
            lambda old, new: jnp.where(active.reshape(active.shape + (1,) * (old.ndim - 1)), new, old),
            states,
            timesteps.last_state,
        )
        memory_zero = jax.tree.map(lambda value: value[:games], memories)
        memory_one = jax.tree.map(lambda value: value[games:], memories)
        newly_done = active & (timesteps.terminated | timesteps.truncated)
        winner = timesteps.info.winner
        actor_won = winner == actor_flat
        actor_lost = (winner >= 0) & ~actor_won
        new_outcome = jnp.where(actor_won, 1.0, jnp.where(actor_lost, 0.0, 0.5))
        outcome = jnp.where(newly_done, new_outcome, outcome)
        relative_turn = scan_index + 2
        finish_relative_turn = jnp.where(newly_done, relative_turn, finish_relative_turn)
        done |= newly_done

        owns_site = states.ownership[flat_index, actor_flat, rows_flat, cols_flat]
        site_is_castle = states.castles[flat_index, rows_flat, cols_flat]
        ever_lost_site |= (branch == BRANCH_BUILD) & ~owns_site
        produced = (
            active & (branch == BRANCH_BUILD) & owns_site & site_is_castle & (states.time % 2 == 0) & (winner < 0)
        )
        production_ticks += produced.astype(jnp.int32)
        newly_paid_back = (payback_relative_turn < 0) & (production_ticks >= cost_flat)
        payback_relative_turn = jnp.where(newly_paid_back, relative_turn, payback_relative_turn)
        for horizon_index, horizon in enumerate(HORIZONS):
            survived_horizons = survived_horizons.at[:, horizon_index].set(
                jnp.where(
                    relative_turn == horizon,
                    ~ever_lost_site,
                    survived_horizons[:, horizon_index],
                )
            )
        return (
            states,
            rng,
            memory_zero,
            memory_one,
            done,
            outcome,
            finish_relative_turn,
            ever_lost_site,
            production_ticks,
            payback_relative_turn,
            survived_horizons,
        ), None

    initial = (
        states,
        key,
        memory_zero,
        memory_one,
        done,
        outcome,
        finish_relative_turn,
        ever_lost_site,
        production_ticks,
        payback_relative_turn,
        survived_horizons,
    )
    final, _ = jax.lax.scan(step, initial, jnp.arange(rollout_steps - 1))
    final_states = final[0]
    final_owns_site = final_states.ownership[flat_index, actor_flat, rows_flat, cols_flat]
    results = {
        "post_actor_value": post_actor_values.reshape(count, repetitions, 2),
        "common_critic_post_actor_values": common_post_actor_values.reshape(
            count, repetitions, 2, len(critic_networks)
        ),
        "intervention_done": done.reshape(count, repetitions, 2),
        "outcome": final[5].reshape(count, repetitions, 2),
        "finished": final[4].reshape(count, repetitions, 2),
        "finish_relative_turn": final[6].reshape(count, repetitions, 2),
        "ever_lost_site": final[7].reshape(count, repetitions, 2),
        "production_ticks": final[8].reshape(count, repetitions, 2),
        "payback_relative_turn": final[9].reshape(count, repetitions, 2),
        "survived_horizons": final[10].reshape(count, repetitions, 2, len(HORIZONS)),
        "owns_site_at_end": final_owns_site.reshape(count, repetitions, 2),
    }
    return features, results


def _tree_flatten_slots(tree):
    return jax.tree.map(
        lambda value: np.asarray(jax.device_get(value)).reshape((-1, *value.shape[3:])),
        tree,
    )


def _tree_take(tree, indices):
    return jax.tree.map(lambda value: jnp.asarray(value[indices]), tree)


def _tree_concat(trees):
    return jax.tree.map(lambda *values: np.concatenate(values), *trees)


def _choose_stratified(found, requested, seed):
    rng = np.random.default_rng(seed)
    flat_found = found.reshape(-1)
    bin_ids = np.broadcast_to(np.arange(TURN_BINS)[None, None, :], found.shape).reshape(-1)
    per_bin = requested // TURN_BINS
    remainder = requested % TURN_BINS
    chosen = []
    leftovers = []
    for turn_bin in range(TURN_BINS):
        candidates = np.flatnonzero(flat_found & (bin_ids == turn_bin))
        rng.shuffle(candidates)
        target = per_bin + (turn_bin < remainder)
        chosen.extend(candidates[:target].tolist())
        leftovers.extend(candidates[target:].tolist())
    if len(chosen) < requested:
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: requested - len(chosen)])
    if len(chosen) < requested:
        raise RuntimeError(f"Only {len(chosen)} sampled legal opportunities were available; requested {requested}")
    chosen = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(chosen)
    return chosen, bin_ids[chosen]


def _bootstrap_mean_ci(values, seed, draws=10_000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(math.ceil(draws / 500)):
        count = min(500, draws - 500 * len(means))
        if count <= 0:
            break
        sample = rng.integers(0, len(values), size=(count, len(values)))
        means.append(values[sample].mean(axis=1))
    bootstrap = np.concatenate(means)
    return [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])]


def _spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    x_rank = np.argsort(np.argsort(x)).astype(np.float64)
    y_rank = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _outcome_counts(values):
    values = np.asarray(values)
    return {
        "wins": int(np.sum(values == 1.0)),
        "draws": int(np.sum(values == 0.5)),
        "losses": int(np.sum(values == 0.0)),
        "score": float(np.mean(values)),
    }


def _segment_summary(name, labels, deltas, control, build):
    rows = []
    labels = np.asarray(labels)
    for label in dict.fromkeys(labels.tolist()):
        mask = labels == label
        values = deltas[mask]
        standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
        rows.append(
            {
                "segment": str(label),
                "states": int(mask.sum()),
                "control_score": float(control[mask].mean()),
                "build_score": float(build[mask].mean()),
                "paired_delta": float(values.mean()),
                "normal_95": (
                    [
                        float(values.mean() - 1.96 * standard_error),
                        float(values.mean() + 1.96 * standard_error),
                    ]
                    if standard_error is not None
                    else None
                ),
            }
        )
    return {"name": name, "rows": rows}


def _fixed_bins(values, edges, labels):
    return np.asarray(labels, dtype=object)[np.clip(np.digitize(values, edges, right=True), 0, len(labels) - 1)]


def summarize(features, results, metadata, seed):
    outcome = results["outcome"]
    control = outcome[:, :, BRANCH_CONTROL]
    build = outcome[:, :, BRANCH_BUILD]
    per_state_control = control.mean(axis=1)
    per_state_build = build.mean(axis=1)
    deltas = per_state_build - per_state_control
    paired_repetitions = build - control
    half = control.shape[1] // 2
    first_delta = (build[:, :half] - control[:, :half]).mean(axis=1)
    second_delta = (build[:, half:] - control[:, half:]).mean(axis=1)
    select_first = first_delta > 0
    select_second = second_delta > 0
    heldout_first_to_second = (
        np.where(select_first[:, None], build[:, half:], control[:, half:]).mean() - control[:, half:].mean()
    )
    heldout_second_to_first = (
        np.where(select_second[:, None], build[:, :half], control[:, :half]).mean() - control[:, :half].mean()
    )

    transitions = {}
    names = {0.0: "loss", 0.5: "draw", 1.0: "win"}
    for control_value, control_name in names.items():
        for build_value, build_name in names.items():
            transitions[f"control_{control_name}_to_build_{build_name}"] = int(
                np.sum((control == control_value) & (build == build_value))
            )

    survival = results["survived_horizons"][:, :, BRANCH_BUILD]
    payback = results["payback_relative_turn"][:, :, BRANCH_BUILD]
    build_probability = np.maximum(features["total_build_probability"], 1e-30)
    log_build_probability = np.log10(build_probability)
    segments = [
        _segment_summary(
            "turn",
            np.asarray([f"{x // 200 * 200}-{x // 200 * 200 + 199}" for x in features["turn"]]),
            deltas,
            per_state_control,
            per_state_build,
        ),
        _segment_summary(
            "site_cost",
            _fixed_bins(features["site_cost"], [35, 40, 46], ["35", "36-40", "41-46", "47+"]),
            deltas,
            per_state_control,
            per_state_build,
        ),
        _segment_summary(
            "post_build_garrison",
            _fixed_bins(
                features["post_build_garrison"],
                [0, 9, 24],
                ["0", "1-9", "10-24", "25+"],
            ),
            deltas,
            per_state_control,
            per_state_build,
        ),
        _segment_summary(
            "army_margin",
            _fixed_bins(
                features["army_margin"],
                [-50, -1, 49],
                ["<=-50", "-49--1", "0-49", "50+"],
            ),
            deltas,
            per_state_control,
            per_state_build,
        ),
        _segment_summary(
            "land_margin",
            _fixed_bins(
                features["land_margin"],
                [-10, -1, 9],
                ["<=-10", "-9--1", "0-9", "10+"],
            ),
            deltas,
            per_state_control,
            per_state_build,
        ),
        _segment_summary(
            "aggregate_build_probability",
            _fixed_bins(
                build_probability,
                [1e-8, 1e-6, 1e-4],
                ["<1e-8", "1e-8--1e-6", "1e-6--1e-4", ">=1e-4"],
            ),
            deltas,
            per_state_control,
            per_state_build,
        ),
        _segment_summary(
            "distance_to_enemy_land_true",
            _fixed_bins(
                features["distance_to_enemy_land_true"],
                [1, 3, 6],
                ["0-1", "2-3", "4-6", "7+"],
            ),
            deltas,
            per_state_control,
            per_state_build,
        ),
    ]

    ordering = np.argsort(deltas)
    detail_indices = np.concatenate([ordering[:25], ordering[-25:]])
    detail = []
    for index in detail_indices:
        row = {}
        for name, value in features.items():
            selected_value = np.asarray(value)[index]
            row[name] = selected_value.item() if selected_value.ndim == 0 else selected_value.tolist()
        row.update(
            {
                "source_batch": int(metadata["source_batch"][index]),
                "source_slot": int(metadata["source_slot"][index]),
                "turn_bin": int(metadata["turn_bin"][index]),
                "control_score": float(per_state_control[index]),
                "build_score": float(per_state_build[index]),
                "paired_delta": float(deltas[index]),
            }
        )
        detail.append(row)

    return {
        "states": int(len(deltas)),
        "repetitions_per_state": int(control.shape[1]),
        "paired_rollouts": int(control.size),
        "branch_rollouts": int(outcome.size),
        "finished_branch_rollouts": int(results["finished"].sum()),
        "outcomes": {
            "control": _outcome_counts(control),
            "forced_build": _outcome_counts(build),
            "paired_score_delta": float(paired_repetitions.mean()),
            "state_cluster_bootstrap_95": _bootstrap_mean_ci(deltas, seed),
            "paired_repetition_standard_error": float(
                paired_repetitions.std(ddof=1) / math.sqrt(paired_repetitions.size)
            ),
            "transitions": transitions,
        },
        "state_level_effects": {
            "positive_estimate": int(np.sum(deltas > 0)),
            "zero_estimate": int(np.sum(deltas == 0)),
            "negative_estimate": int(np.sum(deltas < 0)),
            "delta_median": float(np.median(deltas)),
            "delta_p10": float(np.quantile(deltas, 0.10)),
            "delta_p90": float(np.quantile(deltas, 0.90)),
            "split_half_delta_spearman": _spearman(first_delta, second_delta),
            "split_half_positive_sign_agreement": float(np.mean((first_delta > 0) == (second_delta > 0))),
            "heldout_oracle": {
                "first_half_select_rate": float(select_first.mean()),
                "second_half_uplift_when_selected_by_first": float(heldout_first_to_second),
                "second_half_select_rate": float(select_second.mean()),
                "first_half_uplift_when_selected_by_second": float(heldout_second_to_first),
                "symmetric_uplift": float((heldout_first_to_second + heldout_second_to_first) / 2),
            },
        },
        "castle_mechanics": {
            "uninterrupted_ownership_survival_rate": {
                str(horizon): float(survival[:, :, index].mean()) for index, horizon in enumerate(HORIZONS)
            },
            "owns_site_at_game_end_rate": float(results["owns_site_at_end"][:, :, BRANCH_BUILD].mean()),
            "gross_production_payback_rate": float(np.mean(payback >= 0)),
            "payback_relative_turn_median": (float(np.median(payback[payback >= 0])) if np.any(payback >= 0) else None),
            "gross_castle_production_ticks_mean": float(results["production_ticks"][:, :, BRANCH_BUILD].mean()),
        },
        "policy_alignment": {
            "aggregate_build_probability_geometric_mean": float(10 ** np.mean(log_build_probability)),
            "aggregate_build_probability_median": float(np.median(build_probability)),
            "best_build_rank_median": float(np.median(features["best_build_rank"])),
            "best_build_logit_margin_median": float(np.median(features["best_build_logit_margin"])),
            "spearman_log_build_probability_vs_paired_delta": _spearman(log_build_probability, deltas),
            "spearman_build_logit_margin_vs_paired_delta": _spearman(features["best_build_logit_margin"], deltas),
            "mean_probability_positive_estimate_states": (
                float(build_probability[deltas > 0].mean()) if np.any(deltas > 0) else None
            ),
            "mean_probability_negative_estimate_states": (
                float(build_probability[deltas < 0].mean()) if np.any(deltas < 0) else None
            ),
        },
        "segments": segments,
        "extreme_state_examples": detail,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", choices=("raw", "ema"), default="ema")
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        help=(
            "Optional checkpoint whose stochastic self-play supplies the legal-build "
            "states. Reusing this checkpoint and seed across evaluations creates an "
            "auditable common held-out state set."
        ),
    )
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--source-policy", choices=("raw", "ema"), default="ema")
    parser.add_argument("--expected-iteration", type=int)
    parser.add_argument("--source-games", type=int, default=1024)
    parser.add_argument("--collection-batch-size", type=int, default=512)
    parser.add_argument("--opportunities", type=int, default=2016)
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--pair-batch-size", type=int, default=16)
    parser.add_argument(
        "--critic-manifest",
        type=Path,
        help=(
            "Optional five-participant manifest. Raw critics from the manifest are "
            "evaluated on the target policy's exact intervention successors."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.source_games % args.collection_batch_size:
        parser.error("--source-games must be divisible by --collection-batch-size")
    if args.collection_batch_size % 16:
        parser.error("--collection-batch-size must be divisible by 16")
    if args.repetitions < 2 or args.repetitions % 2:
        parser.error("--repetitions must be a positive even number >= 2")
    if min(args.opportunities, args.pair_batch_size) < 1:
        parser.error("--opportunities and --pair-batch-size must be positive")
    return args


def main():
    args = parse_args()
    started = time.perf_counter()
    config = TrainingConfig.from_toml(args.config)
    raw, ema, iteration, stage_index = load_checkpoint(config, args.checkpoint)
    if args.expected_iteration is not None and iteration != args.expected_iteration:
        raise ValueError(f"Expected checkpoint iteration {args.expected_iteration}, loaded {iteration}")
    if stage_index != len(config.curriculum) - 1:
        raise ValueError(f"Checkpoint stage {stage_index} is not final stage {len(config.curriculum) - 1}")
    network = ema if args.policy == "ema" else raw
    critic_networks = ()
    critic_metadata = []
    if args.critic_manifest is not None:
        critic_payload = json.loads(args.critic_manifest.read_text(encoding="utf-8"))
        loaded_critics = []
        for item in critic_payload["participants"]:
            critic_config = TrainingConfig.from_toml(Path(item["config"]))
            critic_raw, _, critic_iteration, critic_stage = load_checkpoint(critic_config, Path(item["checkpoint"]))
            expected = int(item["iteration"])
            if critic_iteration != expected:
                raise ValueError(f"Critic {item['name']} iteration {critic_iteration}, expected {expected}")
            if (
                critic_config.input_channels != config.input_channels
                or critic_config.observation_schema != config.observation_schema
                or critic_config.temporal_window != config.temporal_window
            ):
                raise ValueError(f"Critic {item['name']} has incompatible inputs")
            loaded_critics.append(critic_raw)
            critic_metadata.append(
                {
                    "name": item["name"],
                    "iteration": critic_iteration,
                    "stage": critic_stage,
                    "checkpoint": item["checkpoint"],
                    "checkpoint_sha256": _sha256(Path(item["checkpoint"])),
                    "policy": "raw",
                }
            )
        critic_networks = tuple(loaded_critics)
    source_checkpoint = args.source_checkpoint or args.checkpoint
    source_config_path = args.source_config or args.config
    source_config = TrainingConfig.from_toml(source_config_path)
    if source_checkpoint == args.checkpoint and source_config_path == args.config:
        source_raw, source_ema = raw, ema
        source_iteration, source_stage_index = iteration, stage_index
    else:
        source_raw, source_ema, source_iteration, source_stage_index = load_checkpoint(source_config, source_checkpoint)
    if source_stage_index != len(source_config.curriculum) - 1:
        raise ValueError(
            f"Source checkpoint stage {source_stage_index} is not final stage {len(source_config.curriculum) - 1}"
        )
    source_network = source_ema if args.source_policy == "ema" else source_raw
    environment = make_environment(
        source_config,
        source_config.curriculum[source_stage_index],
        pool_size=args.collection_batch_size,
    )

    candidate_batches = []
    last_pool = None
    for batch_index in range(args.source_games // args.collection_batch_size):
        batch_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), batch_index)
        pool_key, collection_key = jax.random.split(batch_key)
        pool, _ = environment.reset(pool_key)
        last_pool = pool
        collected = jax.device_get(
            collect_opportunity_batch(
                source_network,
                pool,
                collection_key,
                source_config,
                environment,
            )
        )
        candidate_batches.append(
            {
                "states": _tree_flatten_slots(collected["states"]),
                "memory_zero": _tree_flatten_slots(collected["memory_zero"]),
                "memory_one": _tree_flatten_slots(collected["memory_one"]),
                "found": np.asarray(collected["found"]).reshape(-1),
                "opportunity_steps": np.asarray(collected["opportunity_steps"]).reshape(-1),
            }
        )
        print(
            f"collected source batch {batch_index + 1}/"
            f"{args.source_games // args.collection_batch_size}: "
            f"{int(candidate_batches[-1]['found'].sum())} reservoirs filled",
            flush=True,
        )

    candidates = {
        "states": _tree_concat([batch["states"] for batch in candidate_batches]),
        "memory_zero": _tree_concat([batch["memory_zero"] for batch in candidate_batches]),
        "memory_one": _tree_concat([batch["memory_one"] for batch in candidate_batches]),
        "found": np.concatenate([batch["found"] for batch in candidate_batches]),
        "opportunity_steps": np.concatenate([batch["opportunity_steps"] for batch in candidate_batches]),
    }
    candidate_shape = (
        args.source_games // args.collection_batch_size,
        args.collection_batch_size,
        2,
        TURN_BINS,
    )
    found_shaped = candidates["found"].reshape(candidate_shape)
    flat_found_for_choice = found_shaped.reshape(-1)
    # The helper expects any leading dimensions and a final turn-bin dimension.
    selected_indices, selected_bins = _choose_stratified(found_shaped, args.opportunities, args.seed + 17)
    if not np.all(flat_found_for_choice[selected_indices]):
        raise AssertionError("Selected an unfilled opportunity reservoir")
    selected = {
        "states": _tree_take(candidates["states"], selected_indices),
        "memory_zero": _tree_take(candidates["memory_zero"], selected_indices),
        "memory_one": _tree_take(candidates["memory_one"], selected_indices),
    }
    selected_actor_seats = ((selected_indices // TURN_BINS) % 2).astype(np.int32)
    selected_source_slots = selected_indices % (args.collection_batch_size * 2 * TURN_BINS)
    selected_source_batches = selected_indices // (args.collection_batch_size * 2 * TURN_BINS)
    selected_metadata = {
        "source_batch": selected_source_batches,
        "source_slot": selected_source_slots,
        "turn_bin": selected_bins,
    }
    selected_state_sha256 = _tree_sha256(
        selected["states"],
        selected["memory_zero"],
        selected["memory_one"],
        selected_actor_seats,
        selected_metadata,
    )
    print(
        "selected opportunities by turn bin: "
        + json.dumps(
            {f"{index * 200}-{index * 200 + 199}": int(np.sum(selected_bins == index)) for index in range(TURN_BINS)}
        ),
        flush=True,
    )

    feature_batches = []
    result_batches = []
    metadata_batches = []
    completed = 0
    # Sorting by turn bin lets late-game strata use shorter static scans.
    for turn_bin in range(TURN_BINS):
        bin_selection = np.flatnonzero(selected_bins == turn_bin)
        for start in range(0, len(bin_selection), args.pair_batch_size):
            indices = bin_selection[start : start + args.pair_batch_size]
            valid = len(indices)
            if valid < args.pair_batch_size:
                indices = np.pad(indices, (0, args.pair_batch_size - valid), mode="edge")
            batch_key = jax.random.fold_in(jax.random.PRNGKey(args.seed + 1), completed)
            features, results = evaluate_pair_batch(
                network,
                critic_networks,
                _tree_take(selected["states"], indices),
                _tree_take(selected["memory_zero"], indices),
                _tree_take(selected["memory_one"], indices),
                jnp.asarray(selected_actor_seats[indices]),
                last_pool,
                batch_key,
                config,
                environment,
                repetitions=args.repetitions,
                rollout_steps=config.truncation - turn_bin * TURN_BIN_WIDTH,
            )
            feature_batches.append(
                {name: np.asarray(jax.device_get(value))[:valid] for name, value in features.items()}
            )
            result_batches.append({name: np.asarray(jax.device_get(value))[:valid] for name, value in results.items()})
            metadata_batches.append({name: value[indices][:valid] for name, value in selected_metadata.items()})
            completed += valid
            print(
                f"evaluated {completed}/{args.opportunities} opportunity states "
                f"({completed * args.repetitions} paired continuations)",
                flush=True,
            )

    features = {name: np.concatenate([batch[name] for batch in feature_batches]) for name in feature_batches[0]}
    results = {name: np.concatenate([batch[name] for batch in result_batches]) for name in result_batches[0]}
    metadata = {name: np.concatenate([batch[name] for batch in metadata_batches]) for name in metadata_batches[0]}
    summary = summarize(features, results, metadata, args.seed + 101)
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_iteration": iteration,
        "checkpoint_stage": stage_index,
        "policy": args.policy,
        "config": str(args.config),
        "map_distribution": "final curriculum stage, exact competition generator",
        "source_sampling": {
            "checkpoint": str(source_checkpoint),
            "checkpoint_sha256": _sha256(source_checkpoint),
            "checkpoint_iteration": source_iteration,
            "checkpoint_stage": source_stage_index,
            "config": str(source_config_path),
            "games": args.source_games,
            "policy": f"{args.source_policy} categorical stochastic self-play",
            "selected_state_sha256": selected_state_sha256,
            "reservoir": "one uniform legal opportunity per game, seat, and 200-turn bin",
            "filled_reservoirs": int(candidates["found"].sum()),
            "legal_opportunity_steps_seen": int(candidates["opportunity_steps"].sum()),
            "selected_by_turn_bin": {
                f"{index * 200}-{index * 200 + 199}": int(np.sum(metadata["turn_bin"] == index))
                for index in range(TURN_BINS)
            },
        },
        "counterfactual": {
            "control_intervention": "highest-logit legal non-build action",
            "build_intervention": "highest-logit legal build site",
            "continuation": "checkpoint policy categorical sampling on both seats",
            "pairing": "common categorical random keys across branches",
            "draw_score": 0.5,
            "horizons": list(HORIZONS),
        },
        "common_action_critics": critic_metadata,
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "atlas.json"
    raw_path = args.output_dir / "paired_rollouts.npz"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        raw_path,
        **{f"feature__{name}": value for name, value in features.items()},
        **{f"result__{name}": value for name, value in results.items()},
        **{f"metadata__{name}": value for name, value in metadata.items()},
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"wrote {report_path} and {raw_path}", flush=True)


if __name__ == "__main__":
    main()

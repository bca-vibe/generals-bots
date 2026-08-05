"""Paired-map evaluation against canonical policy and heuristic opponents."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from generals.core.action import DIRECTIONS
from generals.core.game import get_observation

from .actions import CELL_COUNT, MOVE_PLANES, PASS_INDEX, decode_action, legal_action_mask
from .observation import (
    LEGACY_OBSERVATION_SCHEMA,
    augment_observation,
    init_observation_memory,
    temporal_input,
)


def _batched_memory(size: int, pad_to: int, history_size: int, temporal_window: int):
    memory = init_observation_memory(pad_to, history_size, temporal_window)
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (size, *value.shape)), memory)


def _random_action(key, observation, board_mask):
    mask = legal_action_mask(observation, board_mask)
    index = jax.random.categorical(key, jnp.where(mask, 0.0, -1e9))
    return decode_action(index)


def _select_by_seat(observation_zero, observation_one, choose_zero):
    return jax.tree.map(
        lambda zero, one: jnp.where(choose_zero.reshape((-1,) + (1,) * (zero.ndim - 1)), zero, one),
        observation_zero,
        observation_one,
    )


def _empty_behavior(games: int):
    scalar = jnp.zeros((), dtype=jnp.int32)
    return {
        "actions": scalar,
        "moves": scalar,
        "passes": scalar,
        "half_moves": scalar,
        "reinforce_moves": scalar,
        "expansion_moves": scalar,
        "attack_moves": scalar,
        "builds": scalar,
        "successful_builds": scalar,
        "build_opportunity_steps": scalar,
        "dithers": scalar,
        "moves_after_move": scalar,
        "completed_games": scalar,
        "game_length_sum": scalar,
        "terminal_land_margin_sum": scalar,
        "terminal_army_margin_sum": scalar,
        "had_build_opportunity": jnp.zeros((games,), dtype=jnp.bool_),
        "had_build": jnp.zeros((games,), dtype=jnp.bool_),
        "had_successful_build": jnp.zeros((games,), dtype=jnp.bool_),
        "previous_action": jnp.full((games, 5), -1, dtype=jnp.int32),
        "previous_was_move": jnp.zeros((games,), dtype=jnp.bool_),
    }


def _update_behavior_before_step(behavior, observation, legal_mask, action, active):
    games = action.shape[0]
    indices = jnp.arange(games)
    kind = action[:, 0]
    move = active & (kind == 0)
    passed = active & (kind == 1)
    build = active & (kind == 2)
    build_slice = slice(MOVE_PLANES * CELL_COUNT, PASS_INDEX)
    build_opportunity = active & jnp.any(legal_mask[:, build_slice], axis=1)

    source = action[:, 1:3]
    direction = jnp.clip(action[:, 3], 0, DIRECTIONS.shape[0] - 1)
    destination = source + DIRECTIONS[direction]
    destination_row = jnp.clip(destination[:, 0], 0, observation.armies.shape[-2] - 1)
    destination_col = jnp.clip(destination[:, 1], 0, observation.armies.shape[-1] - 1)
    destination_owned = observation.owned_cells[indices, destination_row, destination_col]
    destination_neutral = observation.neutral_cells[indices, destination_row, destination_col]
    destination_opponent = observation.opponent_cells[indices, destination_row, destination_col]

    previous = behavior["previous_action"]
    previous_direction = jnp.clip(previous[:, 3], 0, DIRECTIONS.shape[0] - 1)
    previous_destination = previous[:, 1:3] + DIRECTIONS[previous_direction]
    reverse_move = (
        move
        & behavior["previous_was_move"]
        & jnp.all(source == previous_destination, axis=1)
        & jnp.all(destination == previous[:, 1:3], axis=1)
    )

    updated = dict(behavior)
    updated["actions"] += active.sum()
    updated["moves"] += move.sum()
    updated["passes"] += passed.sum()
    updated["half_moves"] += (move & (action[:, 4] > 0)).sum()
    updated["reinforce_moves"] += (move & destination_owned).sum()
    updated["expansion_moves"] += (move & destination_neutral).sum()
    updated["attack_moves"] += (move & destination_opponent).sum()
    updated["builds"] += build.sum()
    updated["build_opportunity_steps"] += build_opportunity.sum()
    updated["dithers"] += reverse_move.sum()
    updated["moves_after_move"] += (move & behavior["previous_was_move"]).sum()
    updated["had_build_opportunity"] |= build_opportunity
    updated["had_build"] |= build
    updated["previous_action"] = jnp.where(active[:, None], action, previous)
    updated["previous_was_move"] = active & move
    return updated


def _finish_behavior(behavior, newly_finished, info, policy_is_zero):
    own_land = jnp.where(policy_is_zero, info.land[:, 0], info.land[:, 1])
    opponent_land = jnp.where(policy_is_zero, info.land[:, 1], info.land[:, 0])
    own_army = jnp.where(policy_is_zero, info.army[:, 0], info.army[:, 1])
    opponent_army = jnp.where(policy_is_zero, info.army[:, 1], info.army[:, 0])
    updated = dict(behavior)
    updated["completed_games"] += newly_finished.sum()
    updated["game_length_sum"] += jnp.where(newly_finished, info.time, 0).sum()
    updated["terminal_land_margin_sum"] += jnp.where(newly_finished, own_land - opponent_land, 0).sum()
    updated["terminal_army_margin_sum"] += jnp.where(newly_finished, own_army - opponent_army, 0).sum()
    return updated


def _record_successful_builds(behavior, before_castles, after_castles, action, active):
    """Count selected build actions that actually create a castle."""
    games = action.shape[0]
    indices = jnp.arange(games)
    rows = jnp.clip(action[:, 1], 0, before_castles.shape[-2] - 1)
    cols = jnp.clip(action[:, 2], 0, before_castles.shape[-1] - 1)
    created = active & (action[:, 0] == 2) & ~before_castles[indices, rows, cols] & after_castles[indices, rows, cols]
    updated = dict(behavior)
    updated["successful_builds"] += created.sum()
    updated["had_successful_build"] |= created
    return updated


def _behavior_result(behavior, prefix: str = "behavior_"):
    result = {
        f"{prefix}{name}": value
        for name, value in behavior.items()
        if name
        not in {
            "had_build_opportunity",
            "had_build",
            "had_successful_build",
            "previous_action",
            "previous_was_move",
        }
    }
    result[f"{prefix}games_with_build_opportunity"] = behavior["had_build_opportunity"].sum()
    result[f"{prefix}games_with_build"] = behavior["had_build"].sum()
    result[f"{prefix}games_with_successful_build"] = behavior["had_successful_build"].sum()
    return result


def round_robin_matrices(names, matches):
    """Build row-policy matrices from upper-triangular matchup records."""
    size = len(names)
    matrices = {
        name: [[None for _ in range(size)] for _ in range(size)]
        for name in ("score", "win_rate", "castles_built", "games_with_castle")
    }
    index = {name: position for position, name in enumerate(names)}
    for match in matches:
        i, j = index[match["a"]], index[match["b"]]
        if i == j:
            matrices["score"][i][i] = match["score"]
            matrices["win_rate"][i][i] = match["win_rate"]
            matrices["castles_built"][i][i] = 0.5 * (
                match["behavior_a_successful_builds"] + match["behavior_b_successful_builds"]
            )
            matrices["games_with_castle"][i][i] = 0.5 * (
                match["behavior_a_games_with_successful_build"] + match["behavior_b_games_with_successful_build"]
            )
            continue
        matrices["score"][i][j] = match["score"]
        matrices["score"][j][i] = 1.0 - match["score"]
        matrices["win_rate"][i][j] = match["win_rate"]
        matrices["win_rate"][j][i] = match["losses"] / match["games"]
        matrices["castles_built"][i][j] = match["behavior_a_successful_builds"]
        matrices["castles_built"][j][i] = match["behavior_b_successful_builds"]
        matrices["games_with_castle"][i][j] = match["behavior_a_games_with_successful_build"]
        matrices["games_with_castle"][j][i] = match["behavior_b_games_with_successful_build"]
    return {"labels": list(names), **matrices}


def evaluate_paired_vs_opponent(
    environment,
    pool,
    network,
    key,
    n_maps: int,
    truncation: int,
    opponent_action,
    *,
    pad_to: int = 21,
    history_size: int = 7,
    temporal_window: int = 512,
    observation_schema: str = LEGACY_OBSERVATION_SCHEMA,
):
    """Play every selected map twice, swapping the network's player seat.

    ``opponent_action`` may be a plain callable receiving ``(key, observation,
    board_mask)`` or an ``OpponentPolicy`` with functional per-match memory.
    Keeping it explicit lets callers compile one evaluator per opponent while
    sharing map, seat-swap, observation-memory, and result accounting.
    """
    selected = jax.tree.map(lambda value: value[:n_maps], pool)
    states = jax.tree.map(lambda value: jnp.concatenate([value, value]), selected)
    network_is_zero = jnp.arange(2 * n_maps) < n_maps
    memory = _batched_memory(2 * n_maps, pad_to, history_size, temporal_window)
    if hasattr(opponent_action, "initial_memory"):
        single_opponent_memory = opponent_action.initial_memory(pad_to)
        opponent_memory = jax.tree.map(
            lambda value: jnp.broadcast_to(value, (2 * n_maps, *value.shape)),
            single_opponent_memory,
        )
        opponent_step = opponent_action.step
    else:
        opponent_memory = jnp.zeros((2 * n_maps,), dtype=jnp.int32)

        def opponent_step(key, observation, board_mask, current_memory):
            return opponent_action(key, observation, board_mask), current_memory

    games = 2 * n_maps
    finished = jnp.zeros((games,), dtype=jnp.bool_)
    outcomes = jnp.full((2 * n_maps,), 0.5, dtype=jnp.float32)
    behavior = _empty_behavior(games)

    def step(carry, _):
        states, rng, memory, opponent_memory, finished, outcomes, behavior = carry
        observation_zero = jax.vmap(lambda state: get_observation(state, 0))(states)
        observation_one = jax.vmap(lambda state: get_observation(state, 1))(states)
        network_observation = _select_by_seat(observation_zero, observation_one, network_is_zero)
        opponent_observation = _select_by_seat(observation_zero, observation_one, ~network_is_zero)

        augmented, memory = jax.vmap(
            lambda observation, current_memory, board_mask: augment_observation(
                observation,
                current_memory,
                board_mask,
                observation_schema,
                environment.deathtouch_turn or 800,
            )
        )(network_observation, memory, states.board_mask)
        histories = temporal_input(memory)
        masks = jax.vmap(legal_action_mask)(network_observation, states.board_mask)
        network_actions = jax.vmap(
            lambda obs, history, mask: decode_action(jnp.argmax(network.forward(obs, history, mask)[0]))
        )(augmented, histories, masks)
        active = ~finished
        behavior = _update_behavior_before_step(behavior, network_observation, masks, network_actions, active)

        split_keys = jax.random.split(rng, 2 * n_maps + 1)
        rng = split_keys[0]
        opponent_actions, opponent_memory = jax.vmap(opponent_step)(
            split_keys[1:],
            opponent_observation,
            states.board_mask,
            opponent_memory,
        )
        actions_zero = jnp.where(network_is_zero[:, None], network_actions, opponent_actions)
        actions_one = jnp.where(network_is_zero[:, None], opponent_actions, network_actions)
        timesteps, states = jax.vmap(lambda state, actions: environment.step(state, actions, pool))(
            states, jnp.stack([actions_zero, actions_one], axis=1)
        )

        done = timesteps.terminated | timesteps.truncated
        newly_finished = done & ~finished
        network_won = jnp.where(network_is_zero, timesteps.info.winner == 0, timesteps.info.winner == 1)
        network_lost = jnp.where(network_is_zero, timesteps.info.winner == 1, timesteps.info.winner == 0)
        result = jnp.where(network_won, 1.0, jnp.where(network_lost, 0.0, 0.5))
        outcomes = jnp.where(newly_finished, result, outcomes)
        behavior = _finish_behavior(behavior, newly_finished, timesteps.info, network_is_zero)
        finished = finished | done
        return (
            states,
            rng,
            memory,
            opponent_memory,
            finished,
            outcomes,
            behavior,
        ), None

    (_, key, _, _, finished, outcomes, behavior), _ = jax.lax.scan(
        step,
        (states, key, memory, opponent_memory, finished, outcomes, behavior),
        None,
        length=truncation,
    )
    wins = jnp.sum((outcomes == 1.0) & finished)
    losses = jnp.sum((outcomes == 0.0) & finished)
    draws = 2 * n_maps - wins - losses
    paired_scores = (outcomes[:n_maps] + outcomes[n_maps:]) / 2.0
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": outcomes.mean(),
        "paired_score_std": paired_scores.std(),
        **_behavior_result(behavior),
    }, key


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
    sampling: str = "greedy",
    key=None,
):
    """Play paired maps between two neural policies with behavior metrics."""
    if sampling not in {"greedy", "categorical"}:
        raise ValueError("sampling must be 'greedy' or 'categorical'")
    if key is None:
        key = jax.random.PRNGKey(0)
    selected = jax.tree.map(lambda value: value[:n_maps], pool)
    states = jax.tree.map(lambda value: jnp.concatenate([value, value]), selected)
    games = 2 * n_maps
    a_is_zero = jnp.arange(games) < n_maps
    memory_a = _batched_memory(games, pad_to, history_size, temporal_window)
    memory_b = _batched_memory(games, pad_to, history_size, temporal_window)
    finished = jnp.zeros((games,), dtype=jnp.bool_)
    outcomes = jnp.full((games,), 0.5, dtype=jnp.float32)
    behavior_a = _empty_behavior(games)
    behavior_b = _empty_behavior(games)

    def step(carry, _):
        (
            states,
            rng,
            memory_a,
            memory_b,
            finished,
            outcomes,
            behavior_a,
            behavior_b,
        ) = carry
        observation_zero = jax.vmap(lambda state: get_observation(state, 0))(states)
        observation_one = jax.vmap(lambda state: get_observation(state, 1))(states)
        observation_a = _select_by_seat(observation_zero, observation_one, a_is_zero)
        observation_b = _select_by_seat(observation_zero, observation_one, ~a_is_zero)

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
        logits_a = jax.vmap(network_a.forward)(augmented_a, history_a, mask_a)[0]
        logits_b = jax.vmap(network_b.forward)(augmented_b, history_b, mask_b)[0]
        if sampling == "categorical":
            rng, action_key_a, action_key_b = jax.random.split(rng, 3)
            keys_a = jax.random.split(action_key_a, games)
            keys_b = jax.random.split(action_key_b, games)
            indices_a = jax.vmap(jax.random.categorical)(keys_a, logits_a)
            indices_b = jax.vmap(jax.random.categorical)(keys_b, logits_b)
        else:
            indices_a = jnp.argmax(logits_a, axis=-1)
            indices_b = jnp.argmax(logits_b, axis=-1)
        action_a = jax.vmap(decode_action)(indices_a.astype(jnp.int32))
        action_b = jax.vmap(decode_action)(indices_b.astype(jnp.int32))

        active = ~finished
        behavior_a = _update_behavior_before_step(behavior_a, observation_a, mask_a, action_a, active)
        behavior_b = _update_behavior_before_step(behavior_b, observation_b, mask_b, action_b, active)
        prior_castles = states.castles
        actions_zero = jnp.where(a_is_zero[:, None], action_a, action_b)
        actions_one = jnp.where(a_is_zero[:, None], action_b, action_a)
        timesteps, states = jax.vmap(lambda state, actions: environment.step(state, actions, pool))(
            states, jnp.stack([actions_zero, actions_one], axis=1)
        )
        behavior_a = _record_successful_builds(behavior_a, prior_castles, states.castles, action_a, active)
        behavior_b = _record_successful_builds(behavior_b, prior_castles, states.castles, action_b, active)
        done = timesteps.terminated | timesteps.truncated
        newly_finished = done & ~finished
        a_won = jnp.where(a_is_zero, timesteps.info.winner == 0, timesteps.info.winner == 1)
        a_lost = jnp.where(a_is_zero, timesteps.info.winner == 1, timesteps.info.winner == 0)
        result = jnp.where(a_won, 1.0, jnp.where(a_lost, 0.0, 0.5))
        outcomes = jnp.where(newly_finished, result, outcomes)
        behavior_a = _finish_behavior(behavior_a, newly_finished, timesteps.info, a_is_zero)
        behavior_b = _finish_behavior(behavior_b, newly_finished, timesteps.info, ~a_is_zero)
        finished |= done
        return (
            states,
            rng,
            memory_a,
            memory_b,
            finished,
            outcomes,
            behavior_a,
            behavior_b,
        ), None

    final, _ = jax.lax.scan(
        step,
        (
            states,
            key,
            memory_a,
            memory_b,
            finished,
            outcomes,
            behavior_a,
            behavior_b,
        ),
        None,
        length=truncation,
    )
    _, _, _, _, finished, outcomes, behavior_a, behavior_b = final
    wins = jnp.sum((outcomes == 1.0) & finished)
    losses = jnp.sum((outcomes == 0.0) & finished)
    draws = games - wins - losses
    paired_scores = (outcomes[:n_maps] + outcomes[n_maps:]) / 2.0
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": outcomes.mean(),
        "paired_score_std": paired_scores.std(),
        **_behavior_result(behavior_a, "behavior_a_"),
        **_behavior_result(behavior_b, "behavior_b_"),
    }


def evaluate_paired_vs_random(
    environment,
    pool,
    network,
    key,
    n_maps: int,
    truncation: int,
    *,
    pad_to: int = 21,
    history_size: int = 7,
    temporal_window: int = 512,
    observation_schema: str = LEGACY_OBSERVATION_SCHEMA,
):
    """Backward-compatible curriculum evaluator against uniform legal play."""
    return evaluate_paired_vs_opponent(
        environment,
        pool,
        network,
        key,
        n_maps,
        truncation,
        _random_action,
        pad_to=pad_to,
        history_size=history_size,
        temporal_window=temporal_window,
        observation_schema=observation_schema,
    )

"""Paired-map evaluation against canonical policy and heuristic opponents."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from generals.core.game import get_observation

from .actions import decode_action, legal_action_mask
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

    finished = jnp.zeros((2 * n_maps,), dtype=jnp.bool_)
    outcomes = jnp.full((2 * n_maps,), 0.5, dtype=jnp.float32)

    def step(carry, _):
        states, rng, memory, opponent_memory, finished, outcomes = carry
        observation_zero = jax.vmap(lambda state: get_observation(state, 0))(states)
        observation_one = jax.vmap(lambda state: get_observation(state, 1))(states)
        network_observation = jax.tree.map(
            lambda zero, one: jnp.where(
                network_is_zero.reshape((-1,) + (1,) * (zero.ndim - 1)), zero, one
            ),
            observation_zero,
            observation_one,
        )
        opponent_observation = jax.tree.map(
            lambda zero, one: jnp.where(
                network_is_zero.reshape((-1,) + (1,) * (zero.ndim - 1)), one, zero
            ),
            observation_zero,
            observation_one,
        )

        augmented, memory = jax.vmap(
            lambda observation, current_memory, board_mask: augment_observation(
                observation,
                current_memory,
                board_mask,
                observation_schema,
                environment.deathtouch_turn or 800,
            )
        )(
            network_observation, memory, states.board_mask
        )
        histories = temporal_input(memory)
        masks = jax.vmap(legal_action_mask)(network_observation, states.board_mask)
        network_actions = jax.vmap(
            lambda obs, history, mask: decode_action(
                jnp.argmax(network.forward(obs, history, mask)[0])
            )
        )(augmented, histories, masks)

        split_keys = jax.random.split(rng, 2 * n_maps + 1)
        rng = split_keys[0]
        opponent_actions, opponent_memory = jax.vmap(opponent_step)(
            split_keys[1:],
            opponent_observation,
            states.board_mask,
            opponent_memory,
        )
        actions_zero = jnp.where(
            network_is_zero[:, None], network_actions, opponent_actions
        )
        actions_one = jnp.where(
            network_is_zero[:, None], opponent_actions, network_actions
        )
        timesteps, states = jax.vmap(
            lambda state, actions: environment.step(state, actions, pool)
        )(states, jnp.stack([actions_zero, actions_one], axis=1))

        done = timesteps.terminated | timesteps.truncated
        newly_finished = done & ~finished
        network_won = jnp.where(
            network_is_zero, timesteps.info.winner == 0, timesteps.info.winner == 1
        )
        network_lost = jnp.where(
            network_is_zero, timesteps.info.winner == 1, timesteps.info.winner == 0
        )
        result = jnp.where(network_won, 1.0, jnp.where(network_lost, 0.0, 0.5))
        outcomes = jnp.where(newly_finished, result, outcomes)
        finished = finished | done
        return (states, rng, memory, opponent_memory, finished, outcomes), None

    (_, key, _, _, finished, outcomes), _ = jax.lax.scan(
        step,
        (states, key, memory, opponent_memory, finished, outcomes),
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
    }, key


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

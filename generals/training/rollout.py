"""Vectorized symmetric self-play rollout collection."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from generals.core.game import get_observation

from .actions import CELL_COUNT, MOVE_PLANES, PASS_INDEX, decode_action, legal_action_mask
from .castle_exploration import (
    action_distribution_statistics,
    apply_tactical_build_logit_boost,
    tactical_build_mask,
)
from .observation import (
    LEGACY_OBSERVATION_SCHEMA,
    augment_observation,
    reset_finished_memory,
    temporal_input,
)
from .potential import castle_land_potential


def _concatenate_players(player_zero, player_one):
    return jax.tree.map(lambda left, right: jnp.concatenate([left, right]), player_zero, player_one)


def collect_self_play_rollout(
    states,
    pool,
    environment,
    network,
    key,
    memory_player_zero,
    memory_player_one,
    num_steps: int,
    *,
    observation_schema: str = LEGACY_OBSERVATION_SCHEMA,
    potential_shaping_scale=0.0,
    tactical_build_logit_boost=0.0,
    tactical_build_post_reserve: int = 10,
    tactical_build_payback_margin: int = 25,
    tactical_build_remembered_enemy_turns: int = 50,
    tactical_build_enemy_safety_radius: int = 3,
):
    """Collect ``num_steps`` from both seats of every environment.

    Returned arrays have shape ``(time, 2*num_envs, ...)``. The environment
    auto-resets, and the matching deterministic observation memory is reset on
    exactly the same transition.
    """
    num_envs = states.armies.shape[0]
    memory = _concatenate_players(memory_player_zero, memory_player_one)
    deathtouch_turn = environment.deathtouch_turn or 800

    def scan_step(carry, _):
        current_states, rng, current_memory = carry
        obs_zero = jax.vmap(lambda state: get_observation(state, 0))(current_states)
        obs_one = jax.vmap(lambda state: get_observation(state, 1))(current_states)
        observations = _concatenate_players(obs_zero, obs_one)
        board_masks = jnp.concatenate([current_states.board_mask, current_states.board_mask])

        augmented, updated_memory = jax.vmap(
            lambda observation, memory, board_mask: augment_observation(
                observation,
                memory,
                board_mask,
                observation_schema,
                deathtouch_turn,
            )
        )(
            observations, current_memory, board_masks
        )
        histories = temporal_input(updated_memory)
        masks = jax.vmap(legal_action_mask)(observations, board_masks)
        tactical_build_masks = jax.vmap(
            lambda observation, memory, board_mask: tactical_build_mask(
                observation,
                memory,
                board_mask,
                truncation=environment.truncation,
                post_build_reserve=tactical_build_post_reserve,
                payback_margin=tactical_build_payback_margin,
                remembered_enemy_turns=tactical_build_remembered_enemy_turns,
                enemy_safety_radius=tactical_build_enemy_safety_radius,
            )
        )(observations, updated_memory, board_masks)

        split_keys = jax.random.split(rng, 2 * num_envs + 1)
        rng = split_keys[0]

        def sample_policy(observation, history, mask, sample_key, tactical_mask):
            logits, value, _ = network.forward(observation, history, mask)
            behavior_logits = apply_tactical_build_logit_boost(
                logits, tactical_mask, tactical_build_logit_boost
            )
            action_index = jax.random.categorical(
                sample_key, behavior_logits
            ).astype(jnp.int32)
            log_probability = jax.nn.log_softmax(behavior_logits)[action_index]
            return (
                action_index,
                decode_action(action_index),
                value,
                log_probability,
                action_distribution_statistics(logits, mask),
                action_distribution_statistics(behavior_logits, mask),
            )

        (
            action_indices,
            actions,
            values,
            log_probs,
            base_policy_statistics,
            behavior_policy_statistics,
        ) = jax.vmap(sample_policy)(
            augmented,
            histories,
            masks,
            split_keys[1:],
            tactical_build_masks,
        )

        environment_actions = jnp.stack(
            [actions[:num_envs], actions[num_envs:]], axis=1
        )
        timesteps, next_states = jax.vmap(
            lambda state, action: environment.step(state, action, pool)
        )(current_states, environment_actions)
        finished = timesteps.terminated | timesteps.truncated
        updated_memory = reset_finished_memory(
            updated_memory, jnp.concatenate([finished, finished])
        )

        rewards = jnp.concatenate([timesteps.reward[:, 0], timesteps.reward[:, 1]])
        terminated = jnp.concatenate([timesteps.terminated, timesteps.terminated])
        truncated = jnp.concatenate([timesteps.truncated, timesteps.truncated])
        prior_phi = jax.vmap(
            lambda state: castle_land_potential(
                state, truncation=environment.truncation
            )
        )(current_states)
        successor_phi = jax.vmap(
            lambda state, terminated, truncated: castle_land_potential(
                state,
                truncation=environment.truncation,
                terminal=terminated | truncated,
            )
        )(
            timesteps.last_state,
            timesteps.terminated,
            timesteps.truncated,
        )
        potential_rewards_by_game = successor_phi - prior_phi
        potential_rewards = jnp.concatenate(
            [potential_rewards_by_game[:, 0], potential_rewards_by_game[:, 1]]
        )
        actor_rewards = rewards + potential_shaping_scale * potential_rewards
        winners_one = jnp.where(timesteps.info.winner >= 0, 1 - timesteps.info.winner, -1)
        winners = jnp.concatenate([timesteps.info.winner, winners_one])
        build_start = MOVE_PLANES * CELL_COUNT
        selected_build_offset = jnp.clip(
            action_indices - build_start, 0, CELL_COUNT - 1
        )
        selected_tactical_build = (
            (action_indices >= build_start)
            & (action_indices < PASS_INDEX)
            & jnp.take_along_axis(
                tactical_build_masks.reshape(-1, CELL_COUNT),
                selected_build_offset[:, None],
                axis=1,
            )[:, 0]
        )
        data = (
            augmented.astype(jnp.bfloat16),
            histories.astype(jnp.bfloat16),
            masks,
            tactical_build_masks,
            action_indices,
            log_probs,
            values,
            rewards,
            actor_rewards,
            potential_rewards,
            terminated,
            truncated,
            winners,
            base_policy_statistics,
            behavior_policy_statistics,
            selected_tactical_build,
        )
        return (next_states, rng, updated_memory), data

    (states, key, memory), rollout = jax.lax.scan(
        scan_step, (states, key, memory), None, length=num_steps
    )
    (
        observations,
        histories,
        masks,
        tactical_build_masks,
        action_indices,
        log_probs,
        values,
        rewards,
        actor_rewards,
        potential_rewards,
        terminated,
        truncated,
        winners,
        base_policy_statistics,
        behavior_policy_statistics,
        selected_tactical_build,
    ) = rollout

    # Bootstrap only trajectories that remain live at the rollout boundary.
    final_zero = jax.vmap(lambda state: get_observation(state, 0))(states)
    final_one = jax.vmap(lambda state: get_observation(state, 1))(states)
    final_observations = _concatenate_players(final_zero, final_one)
    final_board_masks = jnp.concatenate([states.board_mask, states.board_mask])
    final_augmented, final_memory = jax.vmap(
        lambda observation, current_memory, board_mask: augment_observation(
            observation,
            current_memory,
            board_mask,
            observation_schema,
            deathtouch_turn,
        )
    )(
        final_observations, memory, final_board_masks
    )
    final_histories = temporal_input(final_memory)
    final_masks = jax.vmap(legal_action_mask)(final_observations, final_board_masks)
    final_values = jax.vmap(
        lambda obs, history, mask: network.forward(obs, history, mask)[1]
    )(final_augmented, final_histories, final_masks)
    next_values = jnp.concatenate([values[1:], final_values[None]], axis=0)

    memory_zero = jax.tree.map(lambda value: value[:num_envs], memory)
    memory_one = jax.tree.map(lambda value: value[num_envs:], memory)
    rollout = (
        observations,
        histories,
        masks,
        tactical_build_masks,
        action_indices,
        log_probs,
        values,
        next_values,
        rewards,
        actor_rewards,
        potential_rewards,
        terminated,
        truncated,
        winners,
        base_policy_statistics,
        behavior_policy_statistics,
        selected_tactical_build,
    )
    return states, rollout, key, memory_zero, memory_one

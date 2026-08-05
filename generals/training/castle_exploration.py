"""Tactical, training-only castle exploration utilities.

The competition policy and every evaluator remain unchanged. During the
treatment rollout only, a fixed additive bias is applied to build logits at
sites that satisfy the rule-grounded tactical gate below. The same mask and
bias are replayed during PPO so stored and recomputed log probabilities refer
to exactly the same behavior policy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from generals.core.observation import Observation

from .actions import CELL_COUNT, MOVE_PLANES, PASS_INDEX, build_cost_grid_from_observation
from .observation import ObservationMemory
from .potential import future_growth_events

DEFAULT_POST_BUILD_RESERVE = 10
DEFAULT_PAYBACK_MARGIN = 25
DEFAULT_REMEMBERED_ENEMY_TURNS = 50
DEFAULT_ENEMY_SAFETY_RADIUS = 3


def _dilate_manhattan(mask: jax.Array) -> jax.Array:
    padded = jnp.pad(mask, 1)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )


def _within_manhattan_radius(mask: jax.Array, radius: int) -> jax.Array:
    reached = mask
    for _ in range(radius):
        reached = _dilate_manhattan(reached)
    return reached


def tactical_build_mask(
    observation: Observation,
    memory: ObservationMemory,
    board_mask: jax.Array,
    *,
    truncation: int = 1200,
    post_build_reserve: int = DEFAULT_POST_BUILD_RESERVE,
    payback_margin: int = DEFAULT_PAYBACK_MARGIN,
    remembered_enemy_turns: int = DEFAULT_REMEMBERED_ENEMY_TURNS,
    enemy_safety_radius: int = DEFAULT_ENEMY_SAFETY_RADIUS,
) -> jax.Array:
    """Return eligible 21x21 build sites using policy-available information.

    The gate requires disadvantage, a defensible post-build garrison, no
    visible or recently remembered enemy land within three movement steps,
    and enough actual even-turn production events to repay the live site price
    plus a conservative margin.
    """
    build_cost = build_cost_grid_from_observation(observation)
    plain_owned = (
        observation.owned_cells
        & board_mask
        & ~observation.generals
        & ~observation.castles
    )
    legal = plain_owned & (observation.armies >= build_cost)
    disadvantaged = (
        observation.owned_army_count < observation.opponent_army_count
    ) | (
        observation.owned_land_count - observation.opponent_land_count <= -5
    )
    defensible = observation.armies - build_cost >= post_build_reserve
    remembered_enemy = memory.last_seen_enemy_owned & (
        memory.last_seen_enemy_age <= remembered_enemy_turns
    )
    threatened = _within_manhattan_radius(
        observation.opponent_cells | remembered_enemy,
        enemy_safety_radius,
    )
    production_events = future_growth_events(
        observation.timestep,
        period=2,
        truncation=truncation,
    )
    repays_with_margin = production_events >= build_cost + payback_margin
    return legal & disadvantaged & defensible & ~threatened & repays_with_margin


def apply_tactical_build_logit_boost(
    logits: jax.Array,
    eligible_builds: jax.Array,
    boost: jax.Array | float,
) -> jax.Array:
    """Add a differentiable fixed bias to eligible build-action logits."""
    build_start = MOVE_PLANES * CELL_COUNT
    build_bias = eligible_builds.reshape(-1).astype(logits.dtype) * jnp.asarray(
        boost, dtype=logits.dtype
    )
    bias = jnp.concatenate(
        [
            jnp.zeros((build_start,), dtype=logits.dtype),
            build_bias,
            jnp.zeros((1,), dtype=logits.dtype),
        ]
    )
    return logits + bias


def action_distribution_statistics(
    logits: jax.Array, legal_mask: jax.Array
) -> dict[str, jax.Array]:
    """Summarize action-kind mass and conditional entropies."""
    build_start = MOVE_PLANES * CELL_COUNT
    move_logits = logits[:build_start]
    build_logits = logits[build_start:PASS_INDEX]
    move_legal = legal_mask[:build_start]
    build_legal = legal_mask[build_start:PASS_INDEX]
    pass_logit = logits[PASS_INDEX]
    kind_logits = jnp.stack(
        [
            jax.scipy.special.logsumexp(move_logits),
            jax.scipy.special.logsumexp(build_logits),
            pass_logit,
        ]
    )
    kind_probabilities = jax.nn.softmax(kind_logits)
    kind_log_probabilities = jax.nn.log_softmax(kind_logits)
    kind_entropy = -jnp.sum(kind_probabilities * kind_log_probabilities)

    def conditional_entropy(values, legal):
        probabilities = jax.nn.softmax(values)
        log_probabilities = jax.nn.log_softmax(values)
        entropy = -jnp.sum(probabilities * log_probabilities)
        return jnp.where(jnp.any(legal), entropy, 0.0)

    return {
        "move_probability": kind_probabilities[0],
        "build_probability": kind_probabilities[1],
        "pass_probability": kind_probabilities[2],
        "kind_entropy": kind_entropy,
        "move_conditional_entropy": conditional_entropy(move_logits, move_legal),
        "build_conditional_entropy": conditional_entropy(build_logits, build_legal),
    }

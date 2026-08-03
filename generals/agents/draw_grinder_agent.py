"""Survival-first heuristic intended to force opponents to convert an advantage."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from generals.core.observation import Observation

from .agent import Agent
from .heuristic_utils import (
    PASS_ACTION,
    best_build,
    best_scored_move,
    build_cost_grid,
    destination_values,
    distance_field,
    passable_grid,
    routed_move,
)


class DrawGrinderAgent(Agent):
    """Garrison, intercept only favorable attacks, build, and avoid overextension."""

    def __init__(self, id: str = "DrawGrinder"):
        super().__init__(id)

    def reset(self):
        pass

    @partial(jax.jit, static_argnums=0)
    def act(self, observation: Observation, key: jax.Array) -> jax.Array:
        del key
        obs = observation
        general = obs.generals & obs.owned_cells
        source_army = obs.armies[:, :, None]
        destination_enemy, _ = destination_values(obs.opponent_cells, False)
        destination_general, _ = destination_values(obs.generals & obs.opponent_cells, False)
        destination_army, _ = destination_values(obs.armies, 0)
        can_capture = source_army - 1 > destination_army
        favorable_attack = destination_enemy & can_capture
        attack_scores = (
            source_army.astype(jnp.float32)
            + destination_general.astype(jnp.float32) * 10000.0
            + destination_enemy.astype(jnp.float32) * 100.0
        )
        attack_action = best_scored_move(obs, attack_scores, favorable_attack)

        passable = passable_grid(obs)
        from_general = distance_field(passable, general)
        build_scores = (
            -jnp.abs(from_general.astype(jnp.float32) - 3.0) * 10.0
            - build_cost_grid(obs).astype(jnp.float32)
            + obs.armies.astype(jnp.float32)
        )
        build_action = best_build(obs, build_scores)
        reinforce_action = routed_move(
            obs,
            general,
            source_mask=obs.owned_cells & ~general,
            source_scores=obs.armies.astype(jnp.float32),
        )

        fallback = jnp.where(
            reinforce_action[0] == 0, reinforce_action, PASS_ACTION
        )
        fallback = jnp.where(build_action[0] == 2, build_action, fallback)
        return jnp.where(attack_action[0] == 0, attack_action, fallback)

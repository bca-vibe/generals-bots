"""Economy heuristic that consolidates armies and builds well-spaced castles."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from generals.core.observation import Observation

from .agent import Agent
from .heuristic_utils import (
    best_build,
    best_scored_move,
    build_cost_grid,
    destination_values,
    distance_field,
    neighborhood_count,
    one_hot_cell,
    passable_grid,
)


class CastleEconomistAgent(Agent):
    """Build inexpensive, separated castles and otherwise grow the frontier."""

    def __init__(self, id: str = "CastleEconomist"):
        super().__init__(id)

    def reset(self):
        pass

    @partial(jax.jit, static_argnums=0)
    def act(self, observation: Observation, key: jax.Array) -> jax.Array:
        del key
        obs = observation
        height, width = obs.armies.shape
        passable = passable_grid(obs)
        own_structures = (obs.generals | obs.castles) & obs.owned_cells
        structure_distance = distance_field(jnp.ones_like(passable), own_structures)
        frontier = neighborhood_count(passable & ~obs.owned_cells)
        cost = build_cost_grid(obs)
        plain_owned = obs.owned_cells & ~obs.generals & ~obs.castles

        site_scores = (
            obs.armies.astype(jnp.float32) * 3.0
            - cost.astype(jnp.float32)
            + jnp.minimum(structure_distance, 7).astype(jnp.float32) * 5.0
            + frontier.astype(jnp.float32) * 2.0
        )
        target_index = jnp.argmax(jnp.where(plain_owned, site_scores, -jnp.inf).reshape(-1))
        target = one_hot_cell(target_index, height, width)
        build_action = best_build(obs, site_scores)

        destination_owned, _ = destination_values(obs.owned_cells, False)
        destination_opponent, _ = destination_values(obs.opponent_cells, False)
        destination_army, _ = destination_values(obs.armies, 0)
        destination_target, _ = destination_values(target, False)
        source_army = obs.armies[:, :, None]
        can_capture = source_army - 1 > destination_army
        expansion = ~destination_owned
        desirable = destination_owned | can_capture
        scores = (
            source_army.astype(jnp.float32)
            + 120.0 * (expansion & can_capture)
            + 80.0 * (destination_opponent & can_capture)
            + 180.0 * destination_target
            - 220.0 * target[:, :, None]
        )
        move_action = best_scored_move(obs, scores, desirable)
        return jnp.where(build_action[0] == 2, build_action, move_action)

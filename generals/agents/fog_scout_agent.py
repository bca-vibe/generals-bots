"""Information-seeking heuristic that opens several half-army scouting fronts."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from generals.core.observation import Observation

from .agent import Agent
from .heuristic_utils import (
    best_scored_move,
    destination_values,
    neighborhood_count,
    passable_grid,
)


class FogScoutAgent(Agent):
    """Prefer moves that reveal fog, splitting sufficiently large stacks."""

    def __init__(self, id: str = "FogScout"):
        super().__init__(id)

    def reset(self):
        pass

    @partial(jax.jit, static_argnums=0)
    def act(self, observation: Observation, key: jax.Array) -> jax.Array:
        del key
        obs = observation
        source_army = obs.armies[:, :, None]
        destination_owned, _ = destination_values(obs.owned_cells, False)
        destination_opponent, _ = destination_values(obs.opponent_cells, False)
        destination_general, _ = destination_values(obs.generals & obs.opponent_cells, False)
        destination_army, _ = destination_values(obs.armies, 0)
        reveal_gain, _ = destination_values(neighborhood_count(obs.fog_cells), 0)
        frontier_gain, _ = destination_values(
            neighborhood_count(passable_grid(obs) & ~obs.owned_cells), 0
        )
        split = source_army >= 4
        moving_army = jnp.where(split, source_army // 2, source_army - 1)
        can_capture = moving_army > destination_army
        desirable = destination_owned | can_capture
        scores = (
            source_army.astype(jnp.float32)
            + reveal_gain.astype(jnp.float32) * 40.0
            + frontier_gain.astype(jnp.float32) * 5.0
            + (destination_opponent & can_capture) * 120.0
            + (destination_general & can_capture) * 10000.0
        )
        split = jnp.broadcast_to(split, scores.shape)
        return best_scored_move(obs, scores, desirable, split=split)

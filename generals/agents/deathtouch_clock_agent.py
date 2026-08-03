"""Phase heuristic that consolidates before launching a turn-800 attack."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from generals.core.observation import Observation

from .agent import Agent
from .heuristic_utils import distance_field, one_hot_cell, passable_grid, routed_move


class DeathtouchClockAgent(Agent):
    """Bank a stack early, then hunt aggressively as deathtouch approaches."""

    def __init__(self, id: str = "DeathtouchClock", hunt_turn: int = 760):
        super().__init__(id)
        self.hunt_turn = hunt_turn

    def reset(self):
        pass

    @partial(jax.jit, static_argnums=0)
    def act(self, observation: Observation, key: jax.Array) -> jax.Array:
        del key
        obs = observation
        height, width = obs.armies.shape
        general = obs.generals & obs.owned_cells
        non_general = obs.owned_cells & ~general
        anchor_index = jnp.argmax(
            jnp.where(non_general, obs.armies, -1).reshape(-1)
        )
        anchor = one_hot_cell(anchor_index, height, width) & non_general
        consolidate_action = routed_move(
            obs,
            anchor,
            source_mask=obs.owned_cells & ~anchor & ~general,
            source_scores=obs.armies.astype(jnp.float32),
        )

        passable = passable_grid(obs)
        from_general = distance_field(passable, general)
        fog = obs.fog_cells & passable
        farthest_fog = fog & (from_general == jnp.max(jnp.where(fog, from_general, -1)))
        launch_action = routed_move(
            obs,
            farthest_fog,
            source_mask=general & (obs.armies >= 10),
            split=True,
        )
        prepare_action = jnp.where(
            consolidate_action[0] == 0, consolidate_action, launch_action
        )

        enemy_general = obs.generals & obs.opponent_cells
        enemy_land = obs.opponent_cells
        hunt_goal = jnp.where(
            jnp.any(enemy_general),
            enemy_general,
            jnp.where(jnp.any(enemy_land), enemy_land, farthest_fog),
        )
        hunt_action = routed_move(
            obs,
            hunt_goal,
            source_scores=obs.armies.astype(jnp.float32),
        )
        return jnp.where(obs.timestep >= self.hunt_turn, hunt_action, prepare_action)

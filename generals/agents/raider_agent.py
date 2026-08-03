"""Offensive heuristic that attacks when safe and stages when outmatched."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from generals.core.observation import Observation

from .agent import Agent
from .heuristic_utils import (
    combat_aware_routed_move,
    directions_toward,
    distance_field,
    one_hot_cell,
    passable_grid,
)

GENERAL_RESERVE = 6
DEATHTOUCH_TURN = 800


class RaiderAgent(Agent):
    """Pressure valuable targets, consolidating rather than taking losing fights."""

    def __init__(self, id: str = "Raider"):
        super().__init__(id)

    def reset(self):
        pass

    @partial(jax.jit, static_argnums=0)
    def act(self, observation: Observation, key: jax.Array) -> jax.Array:
        del key
        obs = observation
        enemy_general = obs.generals & obs.opponent_cells
        enemy_castles = obs.castles & obs.opponent_cells
        enemy_land = obs.opponent_cells
        passable = passable_grid(obs)
        own_general = obs.generals & obs.owned_cells
        general_army = jnp.sum(jnp.where(own_general, obs.armies, 0))
        desired_reserve = jnp.maximum(GENERAL_RESERVE, obs.opponent_army_count // 12)
        general_can_leave = general_army >= 2 * desired_reserve
        source_mask = ~own_general | general_can_leave
        from_home = distance_field(passable, own_general)
        reachable_fog = obs.fog_cells & passable & (from_home < obs.armies.size)
        farthest_fog = reachable_fog & (
            from_home == jnp.max(jnp.where(reachable_fog, from_home, -1))
        )
        goals = jnp.where(
            jnp.any(enemy_general),
            enemy_general,
            jnp.where(
                jnp.any(enemy_castles),
                enemy_castles,
                jnp.where(jnp.any(enemy_land), enemy_land, farthest_fog),
            ),
        )
        has_enemy_target = jnp.any(enemy_general | enemy_castles | enemy_land)
        split_sources = (own_general & general_can_leave) | (
            ~has_enemy_target & (obs.armies >= 4)
        )

        to_goal = distance_field(passable, goals)
        _, neighbor_distance = directions_toward(to_goal, passable)
        advances = neighbor_distance < to_goal
        lead_candidates = obs.owned_cells & (obs.armies > 1) & advances & source_mask
        source_scores = (
            obs.armies.astype(jnp.float32) * 4.0 - to_goal.astype(jnp.float32)
        )
        lead_index = jnp.argmax(
            jnp.where(lead_candidates, source_scores, -jnp.inf).reshape(-1)
        )
        lead = one_hot_cell(lead_index, *obs.armies.shape) & lead_candidates

        general_attack_override = (
            own_general
            & jnp.any(enemy_general)
            & ((to_goal == 1) | (obs.timestep >= DEATHTOUCH_TURN))
        )
        attack_split_sources = split_sources & ~(
            own_general & jnp.any(enemy_general) & (to_goal == 1)
        )
        attack_action = combat_aware_routed_move(
            obs,
            goals,
            source_mask=source_mask | general_attack_override,
            source_scores=source_scores,
            split=attack_split_sources,
            allow_destination=enemy_general & (obs.timestep >= DEATHTOUCH_TURN),
        )
        stage_action = combat_aware_routed_move(
            obs,
            lead,
            source_mask=source_mask & ~lead,
            source_scores=obs.armies.astype(jnp.float32),
            split=split_sources,
        )
        return jnp.where(attack_action[0] == 0, attack_action, stage_action)

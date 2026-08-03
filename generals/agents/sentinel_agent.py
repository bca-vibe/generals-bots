"""Defensive heuristic that distinguishes credible threats from visible enemies."""

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
    combat_aware_routed_move,
    destination_values,
    distance_field,
    neighborhood_count,
    one_hot_cell,
    passable_grid,
)

THREAT_RADIUS = 6
CASTLE_RESERVE = 8
MIN_CASTLE_SPACING = 4
DEATHTOUCH_TURN = 800


class SentinelAgent(Agent):
    """Protect the general from credible attacks, then build and expand safely."""

    def __init__(self, id: str = "Sentinel"):
        super().__init__(id)

    def reset(self):
        pass

    @partial(jax.jit, static_argnums=0)
    def act(self, observation: Observation, key: jax.Array) -> jax.Array:
        del key
        obs = observation
        general = obs.generals & obs.owned_cells
        general_army = jnp.sum(jnp.where(general, obs.armies, 0))
        passable = passable_grid(obs)
        from_general = distance_field(passable, general)

        # An enemy is urgent only when its visible army is large enough to
        # project at least two units to the defensive ring within six steps.
        visible_enemy = obs.opponent_cells
        projected_strength = obs.armies - from_general
        credible_threat = (
            visible_enemy
            & (from_general <= THREAT_RADIUS)
            & (projected_strength >= 2)
        )
        strongest_threat = jnp.max(jnp.where(credible_threat, obs.armies, 0))
        desired_garrison = jnp.maximum(12, strongest_threat + 2)
        may_leave_general = general_army >= 2 * desired_garrison
        source_mask = ~general | may_leave_general
        split_sources = general & may_leave_general

        destination_general, _ = destination_values(
            obs.generals & obs.opponent_cells, False
        )
        destination_army, _ = destination_values(obs.armies, 0)
        source_army = obs.armies[:, :, None]
        can_take_general = (source_army - 1 > destination_army) | (
            obs.timestep >= DEATHTOUCH_TURN
        )
        kill_action = best_scored_move(
            obs,
            source_army.astype(jnp.float32),
            candidate_mask=destination_general
            & can_take_general,
        )

        # Interception advances only across friendly cells or positions the
        # selected army can capture. Deathtouch onto the general is exempt.
        threat_action = combat_aware_routed_move(
            obs,
            credible_threat,
            source_mask=source_mask,
            source_scores=obs.armies.astype(jnp.float32),
            split=split_sources,
            allow_destination=(obs.generals & obs.opponent_cells)
            & (obs.timestep >= DEATHTOUCH_TURN),
        )
        screen_action = combat_aware_routed_move(
            obs,
            general,
            source_mask=obs.owned_cells & ~general,
            source_scores=obs.armies.astype(jnp.float32),
        )

        # Reinforce whichever nearby friendly structure is furthest below its
        # reserve: the general wants the dynamic garrison, castles want six.
        structures = obs.owned_cells & (obs.generals | obs.castles)
        defensive_structures = structures & (from_general <= THREAT_RADIUS)
        structure_distance = distance_field(jnp.ones_like(passable), structures)
        structure_requirement = jnp.where(general, desired_garrison, 6)
        underdefended = defensive_structures & (obs.armies < structure_requirement)
        reinforcement_need = structure_requirement - obs.armies
        reinforce_index = jnp.argmax(
            jnp.where(underdefended, reinforcement_need, -1).reshape(-1)
        )
        reinforce_goal = (
            one_hot_cell(reinforce_index, *obs.armies.shape) & underdefended
        )
        reinforce_action = combat_aware_routed_move(
            obs,
            reinforce_goal,
            source_mask=source_mask & ~reinforce_goal,
            source_scores=obs.armies.astype(jnp.float32),
            split=split_sources,
        )

        # Defensive construction preserves a reserve and avoids proximity
        # surcharges instead of spending any pile that can barely afford a build.
        cost = build_cost_grid(obs)
        defensive_ring = -jnp.abs(from_general.astype(jnp.float32) - 4.0) * 10.0
        defensive_ring -= cost.astype(jnp.float32)
        defensive_ring += obs.armies.astype(jnp.float32)
        build_candidate = (
            obs.owned_cells
            & ~structures
            & (structure_distance >= MIN_CASTLE_SPACING)
            & (obs.armies >= cost + CASTLE_RESERVE)
        )
        build_action = best_build(obs, defensive_ring, candidate_mask=build_candidate)

        # Peacetime expansion uses only four-plus armies and splits large piles,
        # preserving local reserves while still claiming safe frontier cells.
        destination_owned, _ = destination_values(obs.owned_cells, False)
        destination_opponent, _ = destination_values(obs.opponent_cells, False)
        cautious_split = source_army >= 8
        moving_army = jnp.where(cautious_split, source_army // 2, source_army - 1)
        can_capture = moving_army > destination_army
        frontier_gain, _ = destination_values(
            neighborhood_count(passable & ~obs.owned_cells), 0
        )
        cautious_scores = (
            source_army.astype(jnp.float32)
            + 30.0 * frontier_gain.astype(jnp.float32)
            + 100.0 * (destination_opponent & can_capture)
        )
        cautious_sources = source_mask & obs.owned_cells & (obs.armies >= 4)
        cautious_mask = (
            (destination_owned | can_capture) & cautious_sources[:, :, None]
        )
        cautious_action = best_scored_move(
            obs,
            cautious_scores,
            cautious_mask,
            split=jnp.broadcast_to(cautious_split, cautious_scores.shape),
        )

        peacetime = jnp.where(build_action[0] == 2, build_action, cautious_action)
        peacetime = jnp.where(
            reinforce_action[0] == 0, reinforce_action, peacetime
        )
        defense = jnp.where(threat_action[0] == 0, threat_action, screen_action)
        result = jnp.where(jnp.any(credible_threat), defense, peacetime)
        return jnp.where(kill_action[0] == 0, kill_action, result)

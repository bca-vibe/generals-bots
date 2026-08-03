"""Boss: hierarchical tactical, economic, and objective-driven heuristic play."""

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
    combat_aware_routed_move,
    destination_values,
    directions_toward,
    distance_field,
    neighborhood_count,
    one_hot_cell,
    passable_grid,
)

DEATHTOUCH_TURN = 800
ECONOMY_CUTOFF = 680
THREAT_RADIUS = 4
MAX_CASTLES = 4
MIN_CASTLE_SPACING = 5
CASTLE_RESERVE = 12
BASE_GARRISON = 4


class BossAgent(Agent):
    """Hybrid planner with tactical overrides, safe routing, and timed economy."""

    def __init__(self, id: str = "Boss"):
        super().__init__(id)

    def reset(self):
        pass

    @partial(jax.jit, static_argnums=0)
    def act(self, observation: Observation, key: jax.Array) -> jax.Array:
        del key
        obs = observation
        armies = obs.armies
        mine = obs.owned_cells
        height, width = armies.shape
        passable = passable_grid(obs)

        general = mine & obs.generals
        enemy_general = obs.opponent_cells & obs.generals
        enemy_castles = obs.opponent_cells & obs.castles
        enemy_land = obs.opponent_cells
        own_structures = mine & (obs.generals | obs.castles)
        general_army = jnp.sum(jnp.where(general, armies, 0))

        from_general = distance_field(passable, general)
        reachable = from_general < armies.size

        # Estimate which visible enemy stacks can project meaningful force into
        # the home region. Only these may preempt the strategic plan.
        projected_enemy = armies - from_general
        credible_threat = (
            enemy_land
            & (from_general <= THREAT_RADIUS)
            & (projected_enemy >= 2)
        )
        strongest_threat = jnp.max(jnp.where(credible_threat, armies, 0))
        fog_pressure = neighborhood_count(obs.fog_cells & reachable)
        general_fog_pressure = jnp.sum(jnp.where(general, fog_pressure, 0))
        desired_garrison = jnp.maximum(
            BASE_GARRISON + jnp.minimum(general_fog_pressure, 4),
            strongest_threat + 3,
        )
        general_can_leave = general_army >= 2 * desired_garrison
        strategic_sources = ~general | general_can_leave

        destination_army, _ = destination_values(armies, 0)
        destination_enemy, _ = destination_values(enemy_land, False)
        destination_general, _ = destination_values(enemy_general, False)
        destination_castle, _ = destination_values(enemy_castles, False)
        source_army = armies[:, :, None]

        # Immediate general wins are lexicographically first and always send the
        # full stack. After turn 800, merely executing the touch is sufficient.
        wins_now = destination_general & (
            (source_army - 1 > destination_army)
            | (obs.timestep >= DEATHTOUCH_TURN)
        )
        win_action = best_scored_move(
            obs,
            source_army.astype(jnp.float32),
            candidate_mask=wins_now,
        )

        # Take favorable local exchanges before committing to a longer route.
        local_split = jnp.broadcast_to(
            (general & general_can_leave)[:, :, None], destination_enemy.shape
        )
        local_force = jnp.where(local_split, source_army // 2, source_army - 1)
        favorable_local = (
            destination_enemy
            & (local_force > destination_army)
            & strategic_sources[:, :, None]
        )
        local_scores = (
            source_army.astype(jnp.float32)
            + destination_army.astype(jnp.float32) * 2.0
            + destination_enemy.astype(jnp.float32) * 200.0
            + destination_castle.astype(jnp.float32) * 2500.0
            + destination_general.astype(jnp.float32) * 100000.0
        )
        local_attack = best_scored_move(
            obs,
            local_scores,
            candidate_mask=favorable_local,
            split=local_split,
        )

        # High-information fog is preferred, especially near visible enemy land.
        reveal_value = neighborhood_count(obs.fog_cells)
        enemy_adjacency = neighborhood_count(enemy_land)
        fog_score = (
            reveal_value.astype(jnp.float32) * 5.0
            + enemy_adjacency.astype(jnp.float32) * 80.0
            + jnp.minimum(from_general, 30).astype(jnp.float32) * 20.0
        )
        fog_candidates = obs.fog_cells & passable & reachable
        fog_index = jnp.argmax(jnp.where(fog_candidates, fog_score, -jnp.inf).reshape(-1))
        fog_goal = one_hot_cell(fog_index, height, width) & fog_candidates

        # General > territory > information. Castles receive a large tactical
        # capture bonus, but do not pull the whole conveyor off its decapitation
        # route merely because one becomes visible.
        objectives = jnp.where(
            jnp.any(enemy_general),
            enemy_general,
            jnp.where(jnp.any(enemy_land), enemy_land, fog_goal),
        )
        to_objective = distance_field(passable, objectives)
        _, next_distance = directions_toward(to_objective, passable)
        advances = next_distance < to_objective
        lead_candidates = mine & (armies > 1) & advances & strategic_sources
        lead_scores = (
            armies.astype(jnp.float32) * 6.0
            - to_objective.astype(jnp.float32) * 2.0
            + obs.castles.astype(jnp.float32) * 10.0
        )
        lead_index = jnp.argmax(
            jnp.where(lead_candidates, lead_scores, -jnp.inf).reshape(-1)
        )
        lead = one_hot_cell(lead_index, height, width) & lead_candidates

        route_split = general & general_can_leave
        adjacent_general = general & jnp.any(enemy_general) & (to_objective == 1)
        route_split &= ~adjacent_general
        general_win_override = general & jnp.any(enemy_general) & (
            (to_objective == 1) | (obs.timestep >= DEATHTOUCH_TURN)
        )
        route_action = combat_aware_routed_move(
            obs,
            objectives,
            source_mask=strategic_sources | general_win_override,
            source_scores=lead_scores,
            split=route_split,
            allow_destination=enemy_general & (obs.timestep >= DEATHTOUCH_TURN),
        )
        feed_action = combat_aware_routed_move(
            obs,
            objectives,
            source_mask=general & general_can_leave,
            source_scores=armies.astype(jnp.float32),
            split=True,
            allow_destination=enemy_general & (obs.timestep >= DEATHTOUCH_TURN),
        )

        # If the lead cannot advance, merge another safe stack into it.
        support_action = combat_aware_routed_move(
            obs,
            lead,
            source_mask=strategic_sources & ~lead,
            source_scores=armies.astype(jnp.float32),
            split=general & general_can_leave,
        )
        routed_objective_action = jnp.where(
            route_action[0] == 0, route_action, support_action
        )
        objective_action = jnp.where(
            feed_action[0] == 0, feed_action, routed_objective_action
        )

        # Castle construction is an economic action, not merely an affordability
        # check: retain a reserve, limit count, space production, and stop before
        # the deathtouch positioning phase.
        build_cost = build_cost_grid(obs)
        structure_distance = distance_field(jnp.ones_like(passable), own_structures)
        frontier = neighborhood_count(passable & ~mine)
        build_candidate = (
            mine
            & ~own_structures
            & (structure_distance >= MIN_CASTLE_SPACING)
            & (armies >= build_cost + CASTLE_RESERVE)
            & (jnp.sum(obs.castles & mine) < MAX_CASTLES)
            & (obs.timestep < ECONOMY_CUTOFF)
            & ~jnp.any(credible_threat)
            & ~jnp.any(enemy_general)
        )
        build_scores = (
            (armies - build_cost).astype(jnp.float32) * 4.0
            + jnp.minimum(structure_distance, 7).astype(jnp.float32) * 12.0
            + frontier.astype(jnp.float32) * 5.0
            - build_cost.astype(jnp.float32)
        )
        build_action = best_build(obs, build_scores, candidate_mask=build_candidate)

        # Local expansion is the final proactive option. Capture calculations use
        # the force that actually moves after a split.
        destination_owned, _ = destination_values(mine, False)
        destination_fog, _ = destination_values(obs.fog_cells, False)
        reveal_destination, _ = destination_values(reveal_value, 0)
        frontier_destination, _ = destination_values(frontier, 0)
        expansion_split = source_army >= 8
        expansion_force = jnp.where(
            expansion_split, source_army // 2, source_army - 1
        )
        expansion_capture = expansion_force > destination_army
        expansion_sources = mine & strategic_sources & (armies >= 4)
        expansion_mask = (
            (destination_owned | expansion_capture)
            & expansion_sources[:, :, None]
        )
        expansion_scores = (
            source_army.astype(jnp.float32)
            + reveal_destination.astype(jnp.float32) * 25.0
            + frontier_destination.astype(jnp.float32) * 8.0
            + destination_fog.astype(jnp.float32) * 40.0
            + (destination_enemy & expansion_capture).astype(jnp.float32) * 250.0
            + (destination_castle & expansion_capture).astype(jnp.float32) * 2500.0
        )
        expansion_action = best_scored_move(
            obs,
            expansion_scores,
            candidate_mask=expansion_mask,
            split=jnp.broadcast_to(expansion_split, expansion_scores.shape),
        )

        # Credible threats receive a safe intercept; if none exists, consolidate
        # toward the general. This branch suppresses economic and scouting moves.
        threat_action = combat_aware_routed_move(
            obs,
            credible_threat,
            source_mask=strategic_sources,
            source_scores=armies.astype(jnp.float32),
            split=general & general_can_leave,
            allow_destination=enemy_general & (obs.timestep >= DEATHTOUCH_TURN),
        )
        screen_action = combat_aware_routed_move(
            obs,
            general,
            source_mask=mine & ~general,
            source_scores=armies.astype(jnp.float32),
        )
        strongest_defender = jnp.max(
            jnp.where(mine & ~general, armies, 0)
        )
        can_intercept = strongest_defender >= strongest_threat + 2
        defense_action = jnp.where(
            can_intercept & (threat_action[0] == 0), threat_action, screen_action
        )

        strategic_action = jnp.where(
            objective_action[0] == 0, objective_action, expansion_action
        )
        economy_action = jnp.where(
            build_action[0] == 2, build_action, strategic_action
        )
        result = jnp.where(
            jnp.any(credible_threat), defense_action, economy_action
        )
        result = jnp.where(local_attack[0] == 0, local_attack, result)
        result = jnp.where(win_action[0] == 0, win_action, result)
        return jnp.where((result[0] >= 0) & (result[0] <= 2), result, PASS_ACTION)

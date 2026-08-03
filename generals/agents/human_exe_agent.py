"""A competition-rules adaptation of EklipZ's Human.exe heuristic policy.

The original bot is a large stateful Python program.  This implementation is a
clean JAX port of its defining ideas: persistent fog beliefs, enemy-general
prediction from emerging territory, tactical overrides, gather/launch cycles,
information-aware expansion, and efficient consolidation.  Castle building and
the deathtouch clock replace the original neutral-city assumptions.

Original MIT-licensed source: https://github.com/EklipZgit/generals-bot
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

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
    directions_toward,
    distance_field,
    neighborhood_count,
    one_hot_cell,
)

DEATHTOUCH_TURN = 800
BUILD_CUTOFF = 680
MAX_CASTLES = 4
MIN_CASTLE_SPACING = 5
CASTLE_RESERVE = 10
GENERAL_RESERVE = 7
THREAT_HORIZON = 7


class HumanExeMemory(NamedTuple):
    """Compact functional belief state carried by JAX evaluation scans."""

    ever_seen: jax.Array
    ever_seen_enemy: jax.Array
    ever_plain: jax.Array
    known_mountains: jax.Array
    known_castles: jax.Array
    known_enemy_general: jax.Array
    previous_enemy_cells: jax.Array
    last_seen_enemy_army: jax.Array
    last_seen_enemy_age: jax.Array
    enemy_origin_score: jax.Array


def init_human_exe_memory(height: int = 21, width: int | None = None) -> HumanExeMemory:
    """Create an empty Human.exe belief state for one match."""
    width = height if width is None else width
    shape = (height, width)
    boolean = jnp.zeros(shape, dtype=jnp.bool_)
    floating = jnp.zeros(shape, dtype=jnp.float32)
    return HumanExeMemory(
        ever_seen=boolean,
        ever_seen_enemy=boolean,
        ever_plain=boolean,
        known_mountains=boolean,
        known_castles=boolean,
        known_enemy_general=boolean,
        previous_enemy_cells=boolean,
        last_seen_enemy_army=floating,
        last_seen_enemy_age=floating,
        enemy_origin_score=floating,
    )


def _dilate(mask: jax.Array) -> jax.Array:
    return neighborhood_count(mask) > 0


def _update_memory(
    observation: Observation,
    board_mask: jax.Array,
    memory: HumanExeMemory,
) -> HumanExeMemory:
    """Update terrain, army, and likely enemy-origin beliefs."""
    visible_plain = (
        board_mask
        & ~observation.fog_cells
        & ~observation.structures_in_fog
        & ~observation.mountains
        & ~observation.castles
    )
    visible = (
        board_mask
        & ~observation.fog_cells
        & ~observation.structures_in_fog
    )
    ever_seen = memory.ever_seen | visible
    ever_plain = memory.ever_plain | visible_plain

    # Competition maps start without castles.  Consequently every structure
    # hidden on the opening observation is a mountain; a later structure on a
    # cell previously known to be plain is a newly built castle.
    opening_structure = observation.structures_in_fog & (observation.timestep <= 1)
    known_mountains = memory.known_mountains | observation.mountains | opening_structure
    inferred_castles = (
        observation.structures_in_fog & ever_plain & ~known_mountains
    )
    known_castles = memory.known_castles | observation.castles | inferred_castles
    known_enemy_general = (
        memory.known_enemy_general
        | (observation.generals & observation.opponent_cells)
    )

    enemy_visible = observation.opponent_cells
    newly_emerged = enemy_visible & ~memory.previous_enemy_cells
    ever_seen_enemy = memory.ever_seen_enemy | _dilate(enemy_visible)
    last_seen_enemy_army = jnp.where(
        enemy_visible,
        observation.armies.astype(jnp.float32),
        memory.last_seen_enemy_army,
    )
    last_seen_enemy_age = jnp.where(
        enemy_visible, 0.0, memory.last_seen_enemy_age + 1.0
    )

    # Human.exe scores fog cells behind each newly visible enemy line as likely
    # origins.  Accumulating this field makes repeated emergence from one region
    # much stronger evidence than a single scouting tile.
    belief_passable = board_mask & ~known_mountains
    emergence_distance = distance_field(belief_passable, newly_emerged)
    emergence_strength = jnp.sum(
        jnp.where(newly_emerged, jnp.maximum(observation.armies, 1), 0)
    ).astype(jnp.float32)
    origin_evidence = (
        jnp.maximum(0, 12 - emergence_distance).astype(jnp.float32)
        * jnp.minimum(1.0 + jnp.log1p(emergence_strength), 5.0)
        * (~ever_seen).astype(jnp.float32)
    )
    enemy_origin_score = memory.enemy_origin_score * 0.997 + origin_evidence

    return HumanExeMemory(
        ever_seen=ever_seen,
        ever_seen_enemy=ever_seen_enemy,
        ever_plain=ever_plain,
        known_mountains=known_mountains,
        known_castles=known_castles,
        known_enemy_general=known_enemy_general,
        previous_enemy_cells=enemy_visible,
        last_seen_enemy_army=last_seen_enemy_army,
        last_seen_enemy_age=last_seen_enemy_age,
        enemy_origin_score=enemy_origin_score,
    )


def _belief_routed_move(
    observation: Observation,
    passable: jax.Array,
    goals: jax.Array,
    *,
    source_mask: jax.Array,
    source_scores: jax.Array,
    split: jax.Array | bool = False,
    allow_suicide_destination: jax.Array | None = None,
) -> jax.Array:
    """Route through remembered terrain while respecting visible combat odds."""
    distances = distance_field(passable, goals)
    direction, neighbor_distance = directions_toward(distances, passable)
    advances = neighbor_distance < distances

    split_grid = jnp.broadcast_to(
        jnp.asarray(split, dtype=jnp.bool_), observation.armies.shape
    )
    moving_army = jnp.where(
        split_grid, observation.armies // 2, observation.armies - 1
    )
    destination_army, _ = destination_values(observation.armies, 0)
    destination_owned, _ = destination_values(observation.owned_cells, False)
    destination_unseen, _ = destination_values(
        observation.fog_cells | observation.structures_in_fog, False
    )
    selected_army = jnp.take_along_axis(
        destination_army, direction[:, :, None], axis=2
    )[:, :, 0]
    selected_owned = jnp.take_along_axis(
        destination_owned, direction[:, :, None], axis=2
    )[:, :, 0]
    selected_unseen = jnp.take_along_axis(
        destination_unseen, direction[:, :, None], axis=2
    )[:, :, 0]
    safe = selected_owned | selected_unseen | (moving_army > selected_army)
    if allow_suicide_destination is not None:
        allowed, _ = destination_values(allow_suicide_destination, False)
        safe |= jnp.take_along_axis(
            allowed, direction[:, :, None], axis=2
        )[:, :, 0]

    movable = (
        observation.owned_cells
        & (observation.armies > 1)
        & source_mask
        & advances
        & safe
    )
    index = jnp.argmax(
        jnp.where(movable, source_scores, -jnp.inf).reshape(-1)
    )
    height, width = observation.armies.shape
    row, col = jnp.divmod(index, width)
    action = jnp.array(
        [0, row, col, direction[row, col], split_grid[row, col]], dtype=jnp.int32
    )
    return jnp.where(jnp.any(goals) & jnp.any(movable), action, PASS_ACTION)


class HumanExeAgent(Agent):
    """Stateful Human.exe-style policy adapted to the competition modifiers."""

    def __init__(self, id: str = "Human.exe"):
        super().__init__(id)

    def reset(self):
        # Evaluation carries state functionally; there is no mutable host state.
        pass

    @staticmethod
    def initial_memory(board_size: int = 21) -> HumanExeMemory:
        return init_human_exe_memory(board_size)

    @partial(jax.jit, static_argnums=0)
    def act_with_memory(
        self,
        observation: Observation,
        key: jax.Array,
        board_mask: jax.Array,
        memory: HumanExeMemory,
    ) -> tuple[jax.Array, HumanExeMemory]:
        del key
        memory = _update_memory(observation, board_mask, memory)
        return self._choose_action(observation, board_mask, memory), memory

    @partial(jax.jit, static_argnums=0)
    def act(self, observation: Observation, key: jax.Array) -> jax.Array:
        """One-shot compatibility path; league evaluation uses ``act_with_memory``."""
        height, width = observation.armies.shape
        memory = init_human_exe_memory(height, width)
        board_mask = jnp.ones_like(observation.armies, dtype=jnp.bool_)
        action, _ = self.act_with_memory(observation, key, board_mask, memory)
        return action

    def _choose_action(
        self,
        obs: Observation,
        board_mask: jax.Array,
        memory: HumanExeMemory,
    ) -> jax.Array:
        armies = obs.armies
        mine = obs.owned_cells
        height, width = armies.shape
        passable = board_mask & ~memory.known_mountains
        general = mine & obs.generals
        enemy_general = obs.opponent_cells & obs.generals
        own_castles = mine & obs.castles
        enemy_castles = obs.opponent_cells & obs.castles
        own_structures = general | own_castles

        from_general = distance_field(passable, general)
        general_army = jnp.sum(jnp.where(general, armies, 0))
        visible_enemy_army = jnp.where(obs.opponent_cells, armies, 0)

        # 1. Tactical king kills and post-800 touches always override planning.
        destination_army, _ = destination_values(armies, 0)
        destination_general, _ = destination_values(enemy_general, False)
        source_army = armies[:, :, None]
        wins_now = destination_general & (
            (source_army - 1 > destination_army)
            | (obs.timestep >= DEATHTOUCH_TURN)
        )
        win_action = best_scored_move(
            obs, source_army.astype(jnp.float32), candidate_mask=wins_now
        )

        # Remember dangerous armies briefly after they disappear into fog.  A
        # stack is credible when it can project useful force into our general or
        # a production structure within the defensive horizon.
        estimated_enemy_army = jnp.where(
            obs.opponent_cells,
            visible_enemy_army.astype(jnp.float32),
            jnp.maximum(0.0, memory.last_seen_enemy_army - 0.5 * memory.last_seen_enemy_age),
        )
        remembered_enemy = (memory.last_seen_enemy_age < 14) & (estimated_enemy_army > 1)
        structure_distance = distance_field(passable, own_structures)
        credible_threat = (
            (obs.opponent_cells | remembered_enemy)
            & ((from_general <= THREAT_HORIZON) | (structure_distance <= 3))
            & (estimated_enemy_army > jnp.minimum(general_army, 5) + from_general / 2)
        )
        threat_index = jnp.argmax(
            jnp.where(
                credible_threat,
                estimated_enemy_army * 10.0 - from_general.astype(jnp.float32),
                -jnp.inf,
            ).reshape(-1)
        )
        threat_goal = one_hot_cell(threat_index, height, width) & credible_threat

        # 2. Infer the enemy general from repeated enemy emergence.  Before any
        # evidence exists, search distant, high-information fog instead.
        plausible_general = (
            board_mask
            & passable
            & ~memory.ever_seen
            & ~mine
            & ~memory.known_castles
        )
        enemy_history_distance = distance_field(
            passable, memory.ever_seen_enemy | obs.opponent_cells
        )
        prediction_score = (
            memory.enemy_origin_score * 20.0
            + jnp.minimum(from_general, 40).astype(jnp.float32)
            + jnp.maximum(0, 12 - enemy_history_distance).astype(jnp.float32) * 5.0
            + neighborhood_count(obs.fog_cells).astype(jnp.float32) * 2.0
        )
        predicted_index = jnp.argmax(
            jnp.where(plausible_general, prediction_score, -jnp.inf).reshape(-1)
        )
        predicted_general = (
            one_hot_cell(predicted_index, height, width) & plausible_general
        )
        known_general = memory.known_enemy_general & passable
        ordinary_enemy_target = obs.opponent_cells & ~enemy_general
        hunt_prediction = (
            (obs.timestep >= 700)
            | (obs.owned_army_count * 3 > obs.opponent_army_count * 4)
        )
        strategic_goal = jnp.where(
            jnp.any(enemy_general),
            enemy_general,
            jnp.where(
                jnp.any(known_general),
                known_general,
                jnp.where(
                    jnp.any(enemy_castles),
                    enemy_castles,
                    jnp.where(
                        jnp.any(ordinary_enemy_target) & ~hunt_prediction,
                        ordinary_enemy_target,
                        predicted_general,
                    ),
                ),
            ),
        )

        to_goal = distance_field(passable, strategic_goal)
        general_reserve = GENERAL_RESERVE + jnp.minimum(
            jnp.sum(jnp.where(general, neighborhood_count(obs.fog_cells), 0)), 5
        )
        general_can_split = general & (armies >= 2 * general_reserve)
        strategic_sources = ~general | general_can_split
        lead_scores = (
            armies.astype(jnp.float32) * 7.0
            - to_goal.astype(jnp.float32) * 2.0
            + own_castles.astype(jnp.float32) * 5.0
        )
        route_action = _belief_routed_move(
            obs,
            passable,
            strategic_goal,
            source_mask=strategic_sources,
            source_scores=lead_scores,
            split=general_can_split,
            allow_suicide_destination=enemy_general & (obs.timestep >= DEATHTOUCH_TURN),
        )

        # 3. Gather peripheral value into the best forward stack.  This is the
        # compact analogue of Human.exe's pruned MST / value-per-turn gathers.
        _, next_goal_distance = directions_toward(to_goal, passable)
        advancing = next_goal_distance < to_goal
        lead_candidates = mine & (armies > 1) & advancing & strategic_sources
        lead_index = jnp.argmax(
            jnp.where(lead_candidates, lead_scores, -jnp.inf).reshape(-1)
        )
        lead = one_hot_cell(lead_index, height, width) & lead_candidates
        to_lead = distance_field(passable, lead)
        gather_density = (
            (armies - 1).astype(jnp.float32) / (to_lead.astype(jnp.float32) + 1.0)
            + own_castles.astype(jnp.float32) * 3.0
        )
        gather_action = _belief_routed_move(
            obs,
            passable,
            lead,
            source_mask=mine & ~lead & (~general | general_can_split),
            source_scores=gather_density,
            split=general_can_split,
        )

        # 4. Favorable visible exchanges beat a long plan, but calculate the
        # actual force sent after a defensive split from the general.
        destination_enemy, _ = destination_values(obs.opponent_cells, False)
        destination_castle, _ = destination_values(enemy_castles, False)
        local_split = jnp.broadcast_to(general_can_split[:, :, None], destination_enemy.shape)
        local_force = jnp.where(local_split, source_army // 2, source_army - 1)
        favorable_local = (
            destination_enemy
            & (local_force > destination_army)
            & strategic_sources[:, :, None]
        )
        local_scores = (
            source_army.astype(jnp.float32)
            + destination_army.astype(jnp.float32) * 3.0
            + destination_castle.astype(jnp.float32) * 3000.0
            + destination_general.astype(jnp.float32) * 100000.0
        )
        local_action = best_scored_move(
            obs,
            local_scores,
            candidate_mask=favorable_local,
            split=local_split,
        )

        # 5. Information-aware expansion is recalculated each turn, matching the
        # original stateless expansion hunts while using exact split strength.
        reveal = neighborhood_count(obs.fog_cells)
        frontier = neighborhood_count(passable & ~mine)
        destination_owned, _ = destination_values(mine, False)
        destination_fog, _ = destination_values(obs.fog_cells, False)
        destination_reveal, _ = destination_values(reveal, 0)
        destination_frontier, _ = destination_values(frontier, 0)
        expansion_split = source_army >= 10
        expansion_force = jnp.where(expansion_split, source_army // 2, source_army - 1)
        expansion_capture = expansion_force > destination_army
        expansion_candidates = (
            mine[:, :, None]
            & strategic_sources[:, :, None]
            & (source_army >= 4)
            & (destination_owned | expansion_capture)
        )
        expansion_scores = (
            source_army.astype(jnp.float32)
            + destination_reveal.astype(jnp.float32) * 18.0
            + destination_frontier.astype(jnp.float32) * 5.0
            + destination_fog.astype(jnp.float32) * 45.0
            + (destination_enemy & expansion_capture).astype(jnp.float32) * 350.0
            + (destination_castle & expansion_capture).astype(jnp.float32) * 3000.0
        )
        expansion_action = best_scored_move(
            obs,
            expansion_scores,
            candidate_mask=expansion_candidates,
            split=jnp.broadcast_to(expansion_split, expansion_scores.shape),
        )

        # 6. Neutral-city planning becomes build-site economics.  Require a
        # positive payback window, spacing, a post-build garrison, and no active
        # defense emergency; stop before the deathtouch positioning phase.
        build_cost = build_cost_grid(obs)
        spacing = distance_field(jnp.ones_like(passable), own_structures)
        remaining_payback_ticks = jnp.maximum(0, BUILD_CUTOFF - obs.timestep) // 2
        build_candidate = (
            mine
            & ~own_structures
            & (spacing >= MIN_CASTLE_SPACING)
            & (armies >= build_cost + CASTLE_RESERVE)
            & (build_cost < remaining_payback_ticks)
            & (jnp.sum(own_castles) < MAX_CASTLES)
            & (obs.timestep < BUILD_CUTOFF)
            & ~jnp.any(credible_threat)
            & ~jnp.any(enemy_general)
        )
        build_scores = (
            (armies - build_cost).astype(jnp.float32) * 5.0
            + jnp.minimum(spacing, 8).astype(jnp.float32) * 12.0
            + frontier.astype(jnp.float32) * 3.0
            - to_goal.astype(jnp.float32)
        )
        build_action = best_build(obs, build_scores, candidate_mask=build_candidate)

        # 7. Defense intercepts the most urgent projected line with a non-general
        # stack. If an intercept is impossible, collapse value back to the king.
        defense_action = _belief_routed_move(
            obs,
            passable,
            threat_goal,
            source_mask=mine & ~general,
            source_scores=armies.astype(jnp.float32) - from_general.astype(jnp.float32),
        )
        fallback_defense = _belief_routed_move(
            obs,
            passable,
            general,
            source_mask=mine & ~general,
            source_scores=armies.astype(jnp.float32),
        )
        defense_action = jnp.where(
            defense_action[0] == 0, defense_action, fallback_defense
        )

        cycle_turn = obs.timestep % 50
        gather_phase = (
            (obs.timestep >= 50)
            & (cycle_turn >= 27)
            & (cycle_turn < 46)
        )
        army_deficit = obs.owned_army_count * 5 < obs.opponent_army_count * 4
        endgame_hunt = obs.timestep >= 700
        has_target_evidence = (
            jnp.any(enemy_general)
            | jnp.any(known_general)
            | jnp.any(enemy_castles)
            | jnp.any(memory.enemy_origin_score > 0)
        )
        search_expand = ~has_target_evidence & (obs.timestep < 650)
        planned_action = jnp.where(
            gather_phase
            & ~army_deficit
            & ~endgame_hunt
            & (gather_action[0] == 0),
            gather_action,
            route_action,
        )
        planned_action = jnp.where(
            search_expand & (expansion_action[0] == 0),
            expansion_action,
            planned_action,
        )
        planned_action = jnp.where(
            planned_action[0] == 0, planned_action, expansion_action
        )
        economic_action = jnp.where(
            build_action[0] == 2, build_action, planned_action
        )
        result = jnp.where(jnp.any(credible_threat), defense_action, economic_action)
        result = jnp.where(local_action[0] == 0, local_action, result)
        result = jnp.where(win_action[0] == 0, win_action, result)
        return jnp.where((result[0] >= 0) & (result[0] <= 2), result, PASS_ACTION)

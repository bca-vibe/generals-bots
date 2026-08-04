"""Standalone NumPy port of :class:`generals.agents.BossAgent`.

The competition worker does not install the engine package, so this module
keeps the Boss policy self-contained.  Its decisions and row-major tie breaks
mirror ``generals/agents/boss_agent.py`` and ``heuristic_utils.py``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

PASS = (1, 0, 0, 0, 0)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))

DEATHTOUCH_TURN = 800
ECONOMY_CUTOFF = 680
THREAT_RADIUS = 4
MAX_CASTLES = 4
MIN_CASTLE_SPACING = 5
CASTLE_RESERVE = 12
BASE_GARRISON = 4

BASE_BUILD_COST = 35
PROXIMITY_PENALTY = 14
PROXIMITY_DECAY = 2
BUILD_RADIUS = 6


def _destination_values(grid, fill_value):
    """Values at each cell's up/down/left/right destination."""
    height, width = grid.shape
    values = np.full((height, width, 4), fill_value, dtype=grid.dtype)
    in_bounds = np.zeros((height, width, 4), dtype=bool)

    values[1:, :, 0] = grid[:-1, :]
    in_bounds[1:, :, 0] = True
    values[:-1, :, 1] = grid[1:, :]
    in_bounds[:-1, :, 1] = True
    values[:, 1:, 2] = grid[:, :-1]
    in_bounds[:, 1:, 2] = True
    values[:, :-1, 3] = grid[:, 1:]
    in_bounds[:, :-1, 3] = True
    return values, in_bounds


def _distance_field(passable, sources):
    """Multi-source shortest-path distance through known-passable cells."""
    height, width = passable.shape
    infinity = height * width + 5
    distance = np.full((height, width), infinity, dtype=np.int32)
    queue = deque()

    for row, col in np.argwhere(sources):
        row = int(row)
        col = int(col)
        distance[row, col] = 0
        queue.append((row, col))

    while queue:
        row, col = queue.popleft()
        next_distance = distance[row, col] + 1
        for dr, dc in DIRECTIONS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width and passable[nr, nc] and next_distance < distance[nr, nc]:
                distance[nr, nc] = next_distance
                queue.append((nr, nc))
    return distance


def _directions_toward(field, passable):
    """First best direction and neighbor distance for every source cell."""
    infinity = field.size + 7
    destination_distance, _ = _destination_values(field, infinity)
    destination_passable, _ = _destination_values(passable, False)
    values = np.where(destination_passable, destination_distance, infinity)
    direction = np.argmin(values, axis=2).astype(np.int32)
    next_distance = np.min(values, axis=2)
    return direction, next_distance


def _neighborhood_count(mask):
    """Count true cells in each centered 3x3 neighborhood."""
    height, width = mask.shape
    padded = np.pad(mask.astype(np.int32), 1)
    result = np.zeros((height, width), dtype=np.int32)
    for row_offset in range(3):
        for col_offset in range(3):
            result += padded[
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
    return result


def _one_hot_cell(index, height, width):
    result = np.zeros((height, width), dtype=bool)
    result.flat[int(index)] = True
    return result


class Agent:
    """Hierarchical tactical, economic, and objective-driven heuristic."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W

    @staticmethod
    def _best_scored_move(ctx, scores, candidate_mask=None, split=False):
        valid = ctx["legal_moves"].copy()
        if candidate_mask is not None:
            valid &= candidate_mask
        if not np.any(valid):
            return PASS

        masked_scores = np.where(valid, scores, -np.inf)
        flat_index = int(np.argmax(masked_scores))
        source_index, direction = divmod(flat_index, 4)
        row, col = divmod(source_index, ctx["width"])
        split_value = np.asarray(split, dtype=np.int32)
        if split_value.ndim:
            split_value = split_value[row, col, direction]
        return (0, row, col, direction, int(split_value))

    @staticmethod
    def _combat_aware_routed_move(
        ctx,
        goals,
        source_mask=None,
        source_scores=None,
        split=False,
        allow_destination=None,
    ):
        distances = _distance_field(ctx["passable"], goals)
        direction, neighbor_distance = _directions_toward(distances, ctx["passable"])
        advances = neighbor_distance < distances

        split_grid = np.broadcast_to(np.asarray(split, dtype=bool), ctx["armies"].shape)
        moving_army = np.where(split_grid, ctx["armies"] // 2, ctx["armies"] - 1)
        destination_armies, _ = _destination_values(ctx["armies"], 0)
        destination_owned, _ = _destination_values(ctx["mine"], False)
        selected_armies = np.take_along_axis(destination_armies, direction[:, :, None], axis=2)[:, :, 0]
        selected_owned = np.take_along_axis(destination_owned, direction[:, :, None], axis=2)[:, :, 0]
        safe = selected_owned | (moving_army > selected_armies)

        if allow_destination is not None:
            destination_allowed, _ = _destination_values(allow_destination, False)
            selected_allowed = np.take_along_axis(destination_allowed, direction[:, :, None], axis=2)[:, :, 0]
            safe |= selected_allowed

        movable = ctx["mine"] & (ctx["armies"] > 1) & advances & safe
        if source_mask is not None:
            movable &= source_mask
        if source_scores is None:
            source_scores = ctx["armies"].astype(np.float32)
        if not np.any(goals) or not np.any(movable):
            return PASS

        index = int(np.argmax(np.where(movable, source_scores, -np.inf)))
        row, col = divmod(index, ctx["width"])
        return (0, row, col, int(direction[row, col]), int(split_grid[row, col]))

    @staticmethod
    def _build_cost_grid(armies, mine, generals, castles):
        del armies
        height, width = mine.shape
        cost = np.full((height, width), BASE_BUILD_COST, dtype=np.int32)
        structures = mine & (generals | castles)
        for structure_row, structure_col in np.argwhere(structures):
            structure_row = int(structure_row)
            structure_col = int(structure_col)
            row_start = max(0, structure_row - BUILD_RADIUS)
            row_end = min(height, structure_row + BUILD_RADIUS + 1)
            col_start = max(0, structure_col - BUILD_RADIUS)
            col_end = min(width, structure_col + BUILD_RADIUS + 1)
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    distance = abs(row - structure_row) + abs(col - structure_col)
                    surcharge = PROXIMITY_PENALTY - PROXIMITY_DECAY * distance
                    if surcharge > 0:
                        cost[row, col] += surcharge
        return cost

    @staticmethod
    def _best_build(ctx, scores, candidate_mask=None):
        legal = ctx["mine"] & ~ctx["generals"] & ~ctx["castles"] & (ctx["armies"] >= ctx["build_cost"])
        if candidate_mask is not None:
            legal &= candidate_mask
        if not np.any(legal):
            return PASS
        index = int(np.argmax(np.where(legal, scores, -np.inf)))
        row, col = divmod(index, ctx["width"])
        return (2, row, col, 0, 0)

    def act(self, obs):
        armies = np.asarray(obs.army_grid, dtype=np.int32)
        types = np.asarray(obs.type_grid, dtype=np.int8)
        owners = np.asarray(obs.owner_grid, dtype=np.int8)
        height, width = armies.shape

        mine = owners == 1
        opponent = owners == 2
        generals = types == 4
        castles = types == 3
        fog = types == 0
        passable = (types != 2) & (types != 5)
        general = mine & generals
        enemy_general = opponent & generals
        enemy_castles = opponent & castles
        own_structures = mine & (generals | castles)
        general_army = int(np.sum(np.where(general, armies, 0)))

        destination_passable, in_bounds = _destination_values(passable, False)
        legal_moves = mine[:, :, None] & (armies > 1)[:, :, None] & in_bounds & destination_passable
        ctx = {
            "height": height,
            "width": width,
            "armies": armies,
            "mine": mine,
            "generals": generals,
            "castles": castles,
            "passable": passable,
            "legal_moves": legal_moves,
        }

        from_general = _distance_field(passable, general)
        reachable = from_general < armies.size
        credible_threat = opponent & (from_general <= THREAT_RADIUS) & (armies - from_general >= 2)
        strongest_threat = int(np.max(np.where(credible_threat, armies, 0)))
        fog_pressure = _neighborhood_count(fog & reachable)
        general_fog_pressure = int(np.sum(np.where(general, fog_pressure, 0)))
        desired_garrison = max(
            BASE_GARRISON + min(general_fog_pressure, 4),
            strongest_threat + 3,
        )
        general_can_leave = general_army >= 2 * desired_garrison
        strategic_sources = ~general | general_can_leave

        destination_army, _ = _destination_values(armies, 0)
        destination_enemy, _ = _destination_values(opponent, False)
        destination_general, _ = _destination_values(enemy_general, False)
        destination_castle, _ = _destination_values(enemy_castles, False)
        source_army = armies[:, :, None]

        wins_now = destination_general & ((source_army - 1 > destination_army) | (obs.turn >= DEATHTOUCH_TURN))
        win_action = self._best_scored_move(ctx, source_army.astype(np.float32), candidate_mask=wins_now)

        local_split = np.broadcast_to((general & general_can_leave)[:, :, None], destination_enemy.shape)
        local_force = np.where(local_split, source_army // 2, source_army - 1)
        favorable_local = destination_enemy & (local_force > destination_army) & strategic_sources[:, :, None]
        local_scores = (
            source_army.astype(np.float32)
            + destination_army.astype(np.float32) * 2.0
            + destination_enemy.astype(np.float32) * 200.0
            + destination_castle.astype(np.float32) * 2500.0
            + destination_general.astype(np.float32) * 100000.0
        )
        local_attack = self._best_scored_move(
            ctx,
            local_scores,
            candidate_mask=favorable_local,
            split=local_split,
        )

        reveal_value = _neighborhood_count(fog)
        enemy_adjacency = _neighborhood_count(opponent)
        fog_score = (
            reveal_value.astype(np.float32) * 5.0
            + enemy_adjacency.astype(np.float32) * 80.0
            + np.minimum(from_general, 30).astype(np.float32) * 20.0
        )
        fog_candidates = fog & passable & reachable
        fog_index = int(np.argmax(np.where(fog_candidates, fog_score, -np.inf)))
        fog_goal = _one_hot_cell(fog_index, height, width) & fog_candidates

        if np.any(enemy_general):
            objectives = enemy_general
        elif np.any(opponent):
            objectives = opponent
        else:
            objectives = fog_goal

        to_objective = _distance_field(passable, objectives)
        _, next_distance = _directions_toward(to_objective, passable)
        advances = next_distance < to_objective
        lead_candidates = mine & (armies > 1) & advances & strategic_sources
        lead_scores = (
            armies.astype(np.float32) * 6.0 - to_objective.astype(np.float32) * 2.0 + castles.astype(np.float32) * 10.0
        )
        lead_index = int(np.argmax(np.where(lead_candidates, lead_scores, -np.inf)))
        lead = _one_hot_cell(lead_index, height, width) & lead_candidates

        route_split = general & general_can_leave
        adjacent_general = general & np.any(enemy_general) & (to_objective == 1)
        route_split &= ~adjacent_general
        general_win_override = general & np.any(enemy_general) & ((to_objective == 1) | (obs.turn >= DEATHTOUCH_TURN))
        deathtouch_destination = enemy_general & (obs.turn >= DEATHTOUCH_TURN)
        route_action = self._combat_aware_routed_move(
            ctx,
            objectives,
            source_mask=strategic_sources | general_win_override,
            source_scores=lead_scores,
            split=route_split,
            allow_destination=deathtouch_destination,
        )
        feed_action = self._combat_aware_routed_move(
            ctx,
            objectives,
            source_mask=general & general_can_leave,
            source_scores=armies.astype(np.float32),
            split=True,
            allow_destination=deathtouch_destination,
        )
        support_action = self._combat_aware_routed_move(
            ctx,
            lead,
            source_mask=strategic_sources & ~lead,
            source_scores=armies.astype(np.float32),
            split=general & general_can_leave,
        )
        routed_objective_action = route_action if route_action[0] == 0 else support_action
        objective_action = feed_action if feed_action[0] == 0 else routed_objective_action

        build_cost = self._build_cost_grid(armies, mine, generals, castles)
        ctx["build_cost"] = build_cost
        structure_distance = _distance_field(np.ones_like(passable), own_structures)
        frontier = _neighborhood_count(passable & ~mine)
        build_candidate = (
            mine
            & ~own_structures
            & (structure_distance >= MIN_CASTLE_SPACING)
            & (armies >= build_cost + CASTLE_RESERVE)
            & (int(np.sum(castles & mine)) < MAX_CASTLES)
            & (obs.turn < ECONOMY_CUTOFF)
            & ~np.any(credible_threat)
            & ~np.any(enemy_general)
        )
        build_scores = (
            (armies - build_cost).astype(np.float32) * 4.0
            + np.minimum(structure_distance, 7).astype(np.float32) * 12.0
            + frontier.astype(np.float32) * 5.0
            - build_cost.astype(np.float32)
        )
        build_action = self._best_build(ctx, build_scores, candidate_mask=build_candidate)

        destination_owned, _ = _destination_values(mine, False)
        destination_fog, _ = _destination_values(fog, False)
        reveal_destination, _ = _destination_values(reveal_value, 0)
        frontier_destination, _ = _destination_values(frontier, 0)
        expansion_split = source_army >= 8
        expansion_force = np.where(expansion_split, source_army // 2, source_army - 1)
        expansion_capture = expansion_force > destination_army
        expansion_sources = mine & strategic_sources & (armies >= 4)
        expansion_mask = (destination_owned | expansion_capture) & expansion_sources[:, :, None]
        expansion_scores = (
            source_army.astype(np.float32)
            + reveal_destination.astype(np.float32) * 25.0
            + frontier_destination.astype(np.float32) * 8.0
            + destination_fog.astype(np.float32) * 40.0
            + (destination_enemy & expansion_capture).astype(np.float32) * 250.0
            + (destination_castle & expansion_capture).astype(np.float32) * 2500.0
        )
        expansion_action = self._best_scored_move(
            ctx,
            expansion_scores,
            candidate_mask=expansion_mask,
            split=np.broadcast_to(expansion_split, expansion_scores.shape),
        )

        threat_action = self._combat_aware_routed_move(
            ctx,
            credible_threat,
            source_mask=strategic_sources,
            source_scores=armies.astype(np.float32),
            split=general & general_can_leave,
            allow_destination=deathtouch_destination,
        )
        screen_action = self._combat_aware_routed_move(
            ctx,
            general,
            source_mask=mine & ~general,
            source_scores=armies.astype(np.float32),
        )
        strongest_defender = int(np.max(np.where(mine & ~general, armies, 0)))
        can_intercept = strongest_defender >= strongest_threat + 2
        defense_action = threat_action if can_intercept and threat_action[0] == 0 else screen_action

        strategic_action = objective_action if objective_action[0] == 0 else expansion_action
        economy_action = build_action if build_action[0] == 2 else strategic_action
        result = defense_action if np.any(credible_threat) else economy_action
        if local_attack[0] == 0:
            result = local_attack
        if win_action[0] == 0:
            result = win_action
        return result if 0 <= result[0] <= 2 else PASS

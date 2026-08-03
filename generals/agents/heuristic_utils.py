"""Shared JAX primitives for deterministic competition heuristic agents."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from generals.core.action import DIRECTIONS
from generals.modifiers.build_castles import (
    BASE_COST,
    PROXIMITY_DECAY,
    PROXIMITY_PENALTY,
)

PASS_ACTION = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
_BUILD_RADIUS = (PROXIMITY_PENALTY - 1) // PROXIMITY_DECAY


def passable_grid(observation):
    """Terrain a fog-aware heuristic is willing to enter."""
    return ~(observation.mountains | observation.structures_in_fog)


def destination_values(grid, fill_value=0):
    """Return values and in-bounds flags for every ``(row, col, direction)``."""
    height, width = grid.shape
    rows = jnp.arange(height)[:, None, None]
    cols = jnp.arange(width)[None, :, None]
    destination_rows = rows + DIRECTIONS[None, None, :, 0]
    destination_cols = cols + DIRECTIONS[None, None, :, 1]
    in_bounds = (
        (destination_rows >= 0)
        & (destination_rows < height)
        & (destination_cols >= 0)
        & (destination_cols < width)
    )
    safe_rows = jnp.clip(destination_rows, 0, height - 1)
    safe_cols = jnp.clip(destination_cols, 0, width - 1)
    values = grid[safe_rows, safe_cols]
    return jnp.where(in_bounds, values, fill_value), in_bounds


def legal_move_mask(observation):
    destination_passable, in_bounds = destination_values(passable_grid(observation), False)
    movable = observation.owned_cells & (observation.armies > 1)
    return movable[:, :, None] & in_bounds & destination_passable


def best_scored_move(observation, scores, candidate_mask=None, split=False):
    """Choose the highest-scoring legal movement action, or pass."""
    valid = legal_move_mask(observation)
    if candidate_mask is not None:
        valid &= candidate_mask
    masked_scores = jnp.where(valid, scores, -jnp.inf)
    flat_index = jnp.argmax(masked_scores.reshape(-1))
    height, width = observation.armies.shape
    source_index, direction = jnp.divmod(flat_index, 4)
    row, col = jnp.divmod(source_index, width)
    has_move = jnp.any(valid)
    split_value = jnp.asarray(split, dtype=jnp.int32)
    if split_value.ndim:
        split_value = split_value[row, col, direction]
    action = jnp.array([0, row, col, direction, split_value], dtype=jnp.int32)
    return jnp.where(has_move, action, PASS_ACTION)


def distance_field(passable, sources):
    """Shortest-path distance from source cells; large means unreachable."""
    height, width = passable.shape
    infinity = jnp.int32(height * width + 5)

    def relax(_, distance):
        neighbors = jnp.minimum(
            jnp.minimum(
                jnp.roll(distance, 1, 0).at[0].set(infinity),
                jnp.roll(distance, -1, 0).at[-1].set(infinity),
            ),
            jnp.minimum(
                jnp.roll(distance, 1, 1).at[:, 0].set(infinity),
                jnp.roll(distance, -1, 1).at[:, -1].set(infinity),
            ),
        )
        return jnp.where(
            sources,
            jnp.int32(0),
            jnp.where(passable, jnp.minimum(distance, neighbors + 1), infinity),
        )

    initial = jnp.where(sources, jnp.int32(0), infinity)
    return jax.lax.fori_loop(0, height * width, relax, initial)


def directions_toward(field, passable):
    """Best direction and neighbor distance for every source cell."""
    infinity = jnp.int32(field.size + 7)

    def shift(array, fill, amount, axis):
        shifted = jnp.roll(array, amount, axis)
        edge = 0 if amount == 1 else -1
        if axis == 0:
            return shifted.at[edge, :].set(fill)
        return shifted.at[:, edge].set(fill)

    values = jnp.stack(
        [
            jnp.where(
                shift(passable, False, amount, axis),
                shift(field, infinity, amount, axis),
                infinity,
            )
            for amount, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1))
        ]
    )
    return jnp.argmin(values, axis=0).astype(jnp.int32), jnp.min(values, axis=0)


def routed_move(observation, goals, source_mask=None, source_scores=None, split=False):
    """Move the best eligible source one step along a shortest path to ``goals``."""
    passable = passable_grid(observation)
    distances = distance_field(passable, goals)
    direction, neighbor_distance = directions_toward(distances, passable)
    advances = neighbor_distance < distances
    movable = observation.owned_cells & (observation.armies > 1) & advances
    if source_mask is not None:
        movable &= source_mask
    if source_scores is None:
        source_scores = observation.armies.astype(jnp.float32)
    index = jnp.argmax(jnp.where(movable, source_scores, -jnp.inf).reshape(-1))
    height, width = observation.armies.shape
    row, col = jnp.divmod(index, width)
    has_move = jnp.any(goals) & jnp.any(movable)
    split_value = jnp.asarray(split, dtype=jnp.int32)
    if split_value.ndim:
        split_value = split_value[row, col]
    action = jnp.array([0, row, col, direction[row, col], split_value], dtype=jnp.int32)
    return jnp.where(has_move, action, PASS_ACTION)


def combat_aware_routed_move(
    observation,
    goals,
    source_mask=None,
    source_scores=None,
    split=False,
    allow_destination=None,
):
    """Route toward goals without entering an unowned cell the move cannot capture."""
    passable = passable_grid(observation)
    distances = distance_field(passable, goals)
    direction, neighbor_distance = directions_toward(distances, passable)
    advances = neighbor_distance < distances

    split_grid = jnp.broadcast_to(jnp.asarray(split, dtype=jnp.bool_), observation.armies.shape)
    moving_army = jnp.where(split_grid, observation.armies // 2, observation.armies - 1)
    destination_armies, _ = destination_values(observation.armies, 0)
    destination_owned, _ = destination_values(observation.owned_cells, False)
    selected_armies = jnp.take_along_axis(
        destination_armies, direction[:, :, None], axis=2
    )[:, :, 0]
    selected_owned = jnp.take_along_axis(
        destination_owned, direction[:, :, None], axis=2
    )[:, :, 0]
    safe = selected_owned | (moving_army > selected_armies)
    if allow_destination is not None:
        destination_allowed, _ = destination_values(allow_destination, False)
        selected_allowed = jnp.take_along_axis(
            destination_allowed, direction[:, :, None], axis=2
        )[:, :, 0]
        safe |= selected_allowed

    movable = observation.owned_cells & (observation.armies > 1) & advances & safe
    if source_mask is not None:
        movable &= source_mask
    if source_scores is None:
        source_scores = observation.armies.astype(jnp.float32)
    index = jnp.argmax(jnp.where(movable, source_scores, -jnp.inf).reshape(-1))
    height, width = observation.armies.shape
    row, col = jnp.divmod(index, width)
    has_move = jnp.any(goals) & jnp.any(movable)
    action = jnp.array(
        [0, row, col, direction[row, col], split_grid[row, col]], dtype=jnp.int32
    )
    return jnp.where(has_move, action, PASS_ACTION)


def build_cost_grid(observation):
    """Exact competition castle price using only the player's observation."""
    structures = (
        (observation.generals | observation.castles) & observation.owned_cells
    ).astype(jnp.int32)
    padded = jnp.pad(structures, _BUILD_RADIUS)
    height, width = structures.shape
    cost = jnp.full((height, width), BASE_COST, dtype=jnp.int32)
    for row_offset in range(-_BUILD_RADIUS, _BUILD_RADIUS + 1):
        for col_offset in range(-_BUILD_RADIUS, _BUILD_RADIUS + 1):
            surcharge = PROXIMITY_PENALTY - PROXIMITY_DECAY * (
                abs(row_offset) + abs(col_offset)
            )
            if surcharge > 0:
                shifted = padded[
                    _BUILD_RADIUS + row_offset : _BUILD_RADIUS + row_offset + height,
                    _BUILD_RADIUS + col_offset : _BUILD_RADIUS + col_offset + width,
                ]
                cost += surcharge * shifted
    return cost


def legal_build_mask(observation):
    plain_owned = (
        observation.owned_cells & ~observation.generals & ~observation.castles
    )
    return plain_owned & (observation.armies >= build_cost_grid(observation))


def best_build(observation, scores, candidate_mask=None):
    """Choose the highest-scoring legal build, optionally within a site mask."""
    legal = legal_build_mask(observation)
    if candidate_mask is not None:
        legal &= candidate_mask
    index = jnp.argmax(jnp.where(legal, scores, -jnp.inf).reshape(-1))
    height, width = observation.armies.shape
    row, col = jnp.divmod(index, width)
    action = jnp.array([2, row, col, 0, 0], dtype=jnp.int32)
    return jnp.where(jnp.any(legal), action, PASS_ACTION)


def neighborhood_count(mask):
    """Number of true cells in each centered 3x3 neighborhood."""
    return jax.lax.reduce_window(
        mask.astype(jnp.int32), 0, jax.lax.add, (3, 3), (1, 1), "SAME"
    )


def one_hot_cell(index, height, width):
    return jnp.arange(height * width).reshape(height, width) == index

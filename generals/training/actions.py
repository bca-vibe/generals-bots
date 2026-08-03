"""Canonical competition action encoding and exact legal-action masks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from generals.core.action import compute_valid_move_mask
from generals.core.observation import Observation
from generals.modifiers.build_castles import build_cost_grid_from_structures

BOARD_SIZE = 21
CELL_COUNT = BOARD_SIZE * BOARD_SIZE
MOVE_PLANES = 8
SPATIAL_PLANES = 9  # eight movement planes plus build
PASS_INDEX = SPATIAL_PLANES * CELL_COUNT
ACTION_COUNT = PASS_INDEX + 1


def build_cost_grid_from_observation(observation: Observation) -> jax.Array:
    """Compute the exact price of building at every cell from legal information."""
    own_structures = (
        observation.generals | observation.castles
    ) & observation.owned_cells
    return build_cost_grid_from_structures(own_structures)


def legal_build_mask(observation: Observation) -> jax.Array:
    plain_owned = observation.owned_cells & ~observation.generals & ~observation.castles
    return plain_owned & (
        observation.armies >= build_cost_grid_from_observation(observation)
    )


def legal_action_mask(
    observation: Observation, board_mask: jax.Array | None = None
) -> jax.Array:
    """Return the canonical boolean action mask with shape ``(3970,)``."""
    if observation.armies.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(
            f"Expected a padded 21x21 observation, got {observation.armies.shape}"
        )
    if board_mask is None:
        board_mask = jnp.ones_like(observation.mountains, dtype=jnp.bool_)
    movement = compute_valid_move_mask(
        observation.armies,
        observation.owned_cells & board_mask,
        observation.mountains | ~board_mask,
    ).transpose(2, 0, 1)
    movement = jnp.concatenate([movement, movement], axis=0).reshape(-1)
    build = legal_build_mask(observation).reshape(-1)
    return jnp.concatenate([movement, build, jnp.ones((1,), dtype=jnp.bool_)])


def decode_action(index: jax.Array) -> jax.Array:
    """Decode a canonical policy index into ``[kind,row,col,direction,split]``."""
    index = index.astype(jnp.int32)
    spatial_plane = index // CELL_COUNT
    position = index % CELL_COUNT
    row, col = position // BOARD_SIZE, position % BOARD_SIZE
    is_pass = index == PASS_INDEX
    is_build = spatial_plane == MOVE_PLANES
    is_half = (spatial_plane >= 4) & (spatial_plane < MOVE_PLANES)
    direction = spatial_plane % 4
    kind = jnp.where(is_pass, 1, jnp.where(is_build, 2, 0))
    return jnp.array(
        [
            kind,
            jnp.where(is_pass, 0, row),
            jnp.where(is_pass, 0, col),
            jnp.where(is_pass | is_build, 0, direction),
            is_half.astype(jnp.int32),
        ],
        dtype=jnp.int32,
    )


def encode_action(action: jax.Array) -> jax.Array:
    """Encode ``[kind,row,col,direction,split]`` into the canonical policy index."""
    kind, row, col, direction, split = action.astype(jnp.int32)
    move_plane = direction + 4 * (split > 0)
    plane = jnp.where(kind == 2, MOVE_PLANES, move_plane)
    spatial_index = plane * CELL_COUNT + row * BOARD_SIZE + col
    return jnp.where(kind == 1, PASS_INDEX, spatial_index).astype(jnp.int32)

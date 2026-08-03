"""Deterministic state augmentation used by the competition transformer."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from generals.core.observation import Observation
from generals.modifiers.build_castles import build_cost_grid_from_structures

LEGACY_OBSERVATION_SCHEMA = "legacy_38"
COMPETITION_OBSERVATION_SCHEMA = "competition_39"
OBSERVATION_SCHEMAS = frozenset(
    (LEGACY_OBSERVATION_SCHEMA, COMPETITION_OBSERVATION_SCHEMA)
)
COMPETITION_RULE_CHANNEL_NAMES = (
    "deathtouch_active",
    "deathtouch_countdown",
    "build_cost",
)
DEATHTOUCH_COUNTDOWN_TURNS = 200.0


def observation_channel_count(observation_schema: str, history_size: int) -> int:
    """Return the spatial input width for one versioned observation schema."""
    if observation_schema not in OBSERVATION_SCHEMAS:
        raise ValueError(
            f"Unknown observation schema {observation_schema!r}; expected one of {sorted(OBSERVATION_SCHEMAS)}"
        )
    base_channels = 24 if observation_schema == LEGACY_OBSERVATION_SCHEMA else 25
    return base_channels + 2 * history_size


class ObservationMemory(NamedTuple):
    own_army_deltas: jax.Array
    enemy_army_deltas: jax.Array
    previous_own_army: jax.Array
    previous_enemy_army: jax.Array
    known_castles: jax.Array
    known_generals: jax.Array
    known_mountains: jax.Array
    ever_plain: jax.Array
    ever_seen: jax.Array
    ever_seen_enemy: jax.Array
    last_seen_enemy_army: jax.Array
    last_seen_enemy_age: jax.Array
    opponent_army_history: jax.Array
    opponent_land_history: jax.Array


def init_observation_memory(
    board_size: int = 21, history_size: int = 7, temporal_window: int = 512
) -> ObservationMemory:
    spatial = (board_size, board_size)
    history = (history_size, *spatial)
    return ObservationMemory(
        own_army_deltas=jnp.zeros(history, dtype=jnp.float32),
        enemy_army_deltas=jnp.zeros(history, dtype=jnp.float32),
        previous_own_army=jnp.zeros(spatial, dtype=jnp.float32),
        previous_enemy_army=jnp.zeros(spatial, dtype=jnp.float32),
        known_castles=jnp.zeros(spatial, dtype=jnp.bool_),
        known_generals=jnp.zeros(spatial, dtype=jnp.bool_),
        known_mountains=jnp.zeros(spatial, dtype=jnp.bool_),
        ever_plain=jnp.zeros(spatial, dtype=jnp.bool_),
        ever_seen=jnp.zeros(spatial, dtype=jnp.bool_),
        ever_seen_enemy=jnp.zeros(spatial, dtype=jnp.bool_),
        last_seen_enemy_army=jnp.zeros(spatial, dtype=jnp.float32),
        last_seen_enemy_age=jnp.zeros(spatial, dtype=jnp.float32),
        opponent_army_history=jnp.zeros((temporal_window,), dtype=jnp.float32),
        opponent_land_history=jnp.zeros((temporal_window,), dtype=jnp.float32),
    )


def _dilate_3x3(mask: jax.Array) -> jax.Array:
    return jax.lax.reduce_window(
        mask.astype(jnp.int32), 0, jax.lax.max, (3, 3), (1, 1), "SAME"
    ).astype(jnp.bool_)


def augment_observation(
    observation: Observation,
    memory: ObservationMemory,
    board_mask: jax.Array | None = None,
    observation_schema: str = LEGACY_OBSERVATION_SCHEMA,
    deathtouch_turn: int = 800,
) -> tuple[jax.Array, ObservationMemory]:
    """Create a versioned spatial input and update deterministic episode memory."""
    if observation_schema not in OBSERVATION_SCHEMAS:
        raise ValueError(
            f"Unknown observation schema {observation_schema!r}; expected one of {sorted(OBSERVATION_SCHEMAS)}"
        )
    if board_mask is None:
        board_mask = jnp.ones_like(observation.armies, dtype=jnp.bool_)
    padding = ~board_mask
    armies = observation.armies.astype(jnp.float32)
    own_army = armies * observation.owned_cells
    enemy_army = armies * observation.opponent_cells

    own_delta = own_army - memory.previous_own_army
    enemy_delta = enemy_army - memory.previous_enemy_army
    own_deltas = jnp.concatenate([own_delta[None], memory.own_army_deltas[:-1]])
    enemy_deltas = jnp.concatenate([enemy_delta[None], memory.enemy_army_deltas[:-1]])

    visible = (
        ~observation.fog_cells & ~observation.structures_in_fog & board_mask
    ) | padding
    ever_seen = memory.ever_seen | visible
    ever_seen_enemy = memory.ever_seen_enemy | _dilate_3x3(observation.opponent_cells)
    # The interface distinguishes ordinary fog from structures in fog. Therefore
    # an ordinary fog cell proves that no mountain or castle occupies it even
    # though its ownership and army are hidden. Mountains are static, so if a
    # cell ever known to be plain later becomes a fogged structure, the build
    # mechanic implies that it is a castle.
    plain_now = (
        board_mask
        & ~observation.mountains
        & ~observation.castles
        & ~observation.structures_in_fog
    )
    ever_plain = memory.ever_plain | plain_now
    known_generals = memory.known_generals | observation.generals
    known_mountains = memory.known_mountains | observation.mountains | padding
    if observation_schema == COMPETITION_OBSERVATION_SCHEMA:
        # Competition maps contain no castles at game start. Every fogged
        # structure without prior plain evidence is therefore a static
        # mountain. Conversely, ordinary fog is positive plain evidence, so a
        # structure that later appears there can only be a newly built castle.
        initial_fogged_mountains = (
            observation.structures_in_fog & ~memory.ever_plain & ~memory.known_castles
        )
        known_mountains = known_mountains | initial_fogged_mountains
    inferred_castles = (
        observation.structures_in_fog & memory.ever_plain & ~known_mountains
    )
    known_castles = memory.known_castles | observation.castles
    if observation_schema == COMPETITION_OBSERVATION_SCHEMA:
        known_castles = known_castles | inferred_castles

    # Ownership, not positive army count, defines visibility. A newly built
    # castle may legally have zero army and must still reset this memory.
    enemy_visible = observation.opponent_cells
    last_seen_enemy_army = jnp.where(
        enemy_visible, enemy_army, memory.last_seen_enemy_army
    )
    last_seen_enemy_age = jnp.where(
        enemy_visible, 0.0, memory.last_seen_enemy_age + 1.0
    )

    opponent_army_history = (
        jnp.roll(memory.opponent_army_history, -1)
        .at[-1]
        .set(observation.opponent_army_count)
    )
    opponent_land_history = (
        jnp.roll(memory.opponent_land_history, -1)
        .at[-1]
        .set(observation.opponent_land_count)
    )

    height, width = observation.armies.shape
    x_coord = jnp.broadcast_to(
        jnp.arange(width, dtype=jnp.float32)[None] / (width - 1), (height, width)
    )
    y_coord = jnp.broadcast_to(
        jnp.arange(height, dtype=jnp.float32)[:, None] / (height - 1), (height, width)
    )
    ones = jnp.ones((height, width), dtype=jnp.float32)

    common_channels = [
        armies,
        own_army,
        enemy_army,
        ever_seen,
        ever_seen_enemy,
        known_generals,
        known_castles,
        known_mountains,
        observation.neutral_cells,
        observation.owned_cells,
        observation.opponent_cells,
        observation.fog_cells & board_mask,
        observation.timestep * ones,
        (observation.timestep % 50) * ones / 50.0,
        observation.owned_land_count * ones,
        observation.owned_army_count * ones,
        observation.opponent_land_count * ones,
        observation.opponent_army_count * ones,
        last_seen_enemy_army,
        jnp.log1p(last_seen_enemy_age) / 5.0,
        x_coord,
        y_coord,
    ]
    if observation_schema == LEGACY_OBSERVATION_SCHEMA:
        common_channels.insert(3, armies * observation.neutral_cells)
        common_channels.insert(13, observation.structures_in_fog & board_mask)
    else:
        common_channels.extend(competition_rule_channels(observation, deathtouch_turn))
    channels = jnp.stack(
        common_channels,
        axis=0,
        dtype=jnp.float32,
    )
    augmented = jnp.concatenate([channels, own_deltas, enemy_deltas], axis=0)
    new_memory = ObservationMemory(
        own_army_deltas=own_deltas,
        enemy_army_deltas=enemy_deltas,
        previous_own_army=own_army,
        previous_enemy_army=enemy_army,
        known_castles=known_castles,
        known_generals=known_generals,
        known_mountains=known_mountains,
        ever_plain=ever_plain,
        ever_seen=ever_seen,
        ever_seen_enemy=ever_seen_enemy,
        last_seen_enemy_army=last_seen_enemy_army,
        last_seen_enemy_age=last_seen_enemy_age,
        opponent_army_history=opponent_army_history,
        opponent_land_history=opponent_land_history,
    )
    return augmented, new_memory


def reset_finished_memory(
    memory: ObservationMemory, finished: jax.Array
) -> ObservationMemory:
    """Zero batched memory leaves for environments that auto-reset."""

    def reset_leaf(leaf):
        condition = finished.reshape(finished.shape + (1,) * (leaf.ndim - 1))
        return jnp.where(condition, jnp.zeros_like(leaf), leaf)

    return jax.tree.map(reset_leaf, memory)


def temporal_input(memory: ObservationMemory) -> jax.Array:
    return jnp.stack(
        [memory.opponent_army_history, memory.opponent_land_history], axis=-2
    )


def normalize_augmented_observation(
    observation: jax.Array,
    observation_schema: str = LEGACY_OBSERVATION_SCHEMA,
) -> jax.Array:
    """Apply AverageJoe's scale-50 normalization to a versioned input."""
    if observation_schema == LEGACY_OBSERVATION_SCHEMA:
        scaled_channels = jnp.array(
            [0, 1, 2, 3, 14, 16, 17, 18, 19, 20, *range(24, observation.shape[0])]
        )
    elif observation_schema == COMPETITION_OBSERVATION_SCHEMA:
        scaled_channels = jnp.array(
            [0, 1, 2, 12, 14, 15, 16, 17, 18, 24, *range(25, observation.shape[0])]
        )
    else:
        raise ValueError(
            f"Unknown observation schema {observation_schema!r}; expected one of {sorted(OBSERVATION_SCHEMAS)}"
        )
    return observation.at[scaled_channels].divide(50.0)


def competition_rule_channels(
    observation: Observation, deathtouch_turn: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return deathtouch phase, countdown, and exact castle price maps."""
    timestep = observation.timestep.astype(jnp.float32)
    shape = observation.armies.shape
    active = jnp.broadcast_to((timestep >= deathtouch_turn).astype(jnp.float32), shape)
    countdown = jnp.broadcast_to(
        jnp.clip(
            (jnp.asarray(deathtouch_turn, dtype=jnp.float32) - timestep)
            / DEATHTOUCH_COUNTDOWN_TURNS,
            0.0,
            1.0,
        ),
        shape,
    )
    own_structures = (
        observation.generals | observation.castles
    ) & observation.owned_cells
    build_cost = build_cost_grid_from_structures(own_structures).astype(jnp.float32)
    return active, countdown, build_cost

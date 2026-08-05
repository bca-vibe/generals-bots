"""Potential-based reward shaping for the castle exploration experiment.

The potential is deliberately separate from the environment's terminal reward.
It is bounded, zero-sum, and set to zero at every terminal or truncated state so
``r' = r + gamma * Phi(s') - Phi(s)`` telescopes exactly when ``gamma == 1``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from generals.core.game import GameState

LAND_POTENTIAL_SCALE = 0.05
CASTLE_POTENTIAL_SCALE = 0.05
LAND_MARGIN_TEMPERATURE = 20.0
CASTLE_ASSET_TEMPERATURE = 0.10
ARMY_NEED_TEMPERATURE = 50.0
LAND_NEED_TEMPERATURE = 10.0
MINIMUM_CASTLE_NEED = 0.25
GARRISON_CENTER = 10.0
GARRISON_TEMPERATURE = 10.0
CASTLE_HORIZON_EVENTS = 200.0
LAND_HORIZON_EVENTS = 8.0
ENEMY_SAFETY_RADIUS = 6


def _dilate_manhattan(mask: jax.Array) -> jax.Array:
    """Grow a boolean mask by one four-neighbour Manhattan step."""
    padded = jnp.pad(mask, 1)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )


def _capped_enemy_distance(enemy_land: jax.Array) -> jax.Array:
    """Distance to enemy land, capped at six cells.

    Only the range used by the safety term is computed. Cells farther than six
    steps receive distance six, which is already the maximum-safety value.
    """
    reached = enemy_land
    distance = jnp.full(enemy_land.shape, ENEMY_SAFETY_RADIUS, dtype=jnp.int32)
    distance = jnp.where(enemy_land, 0, distance)
    for step in range(1, ENEMY_SAFETY_RADIUS):
        expanded = _dilate_manhattan(reached)
        newly_reached = expanded & ~reached
        distance = jnp.where(newly_reached, step, distance)
        reached = expanded
    return distance


def future_growth_events(
    time: jax.Array,
    *,
    period: int,
    truncation: int,
) -> jax.Array:
    """Count future growth ticks that can affect a nonterminal outcome.

    Growth is applied after the state time advances. The growth event on the
    hard truncation tick itself cannot affect the result, so events are counted
    strictly after ``time`` and strictly before ``truncation``.
    """
    time = jnp.asarray(time, dtype=jnp.int32)
    last_live_tick = jnp.asarray(truncation - 1, dtype=jnp.int32)
    return jnp.maximum(last_live_tick // period - time // period, 0)


def _player_castle_asset(
    state: GameState,
    player: int,
    *,
    truncation: int,
) -> jax.Array:
    """Risk-adjusted economic asset value of one player's current castles."""
    opponent = 1 - player
    own_land = jnp.sum(state.ownership[player])
    enemy_land = jnp.sum(state.ownership[opponent])
    own_army = jnp.sum(state.armies * state.ownership[player])
    enemy_army = jnp.sum(state.armies * state.ownership[opponent])

    # Castles were most useful in the atlas when the builder was behind. This
    # smooth factor gives the same physical castle more marginal asset value in
    # that regime without ever directly rewarding the build action itself.
    raw_need = jax.nn.sigmoid(
        -(own_army - enemy_army) / ARMY_NEED_TEMPERATURE
        - (own_land - enemy_land) / LAND_NEED_TEMPERATURE
    )
    need = MINIMUM_CASTLE_NEED + (1.0 - MINIMUM_CASTLE_NEED) * raw_need

    garrison = jax.nn.sigmoid(
        (state.armies.astype(jnp.float32) - GARRISON_CENTER)
        / GARRISON_TEMPERATURE
    )
    enemy_distance = _capped_enemy_distance(state.ownership[opponent])
    safety = jnp.clip(
        (enemy_distance.astype(jnp.float32) - 1.0)
        / (ENEMY_SAFETY_RADIUS - 1.0),
        0.0,
        1.0,
    )
    growth_events = future_growth_events(
        state.time, period=2, truncation=truncation
    ).astype(jnp.float32)
    horizon = jnp.clip(growth_events / CASTLE_HORIZON_EVENTS, 0.0, 1.0)
    owned_castles = state.castles & state.ownership[player] & state.board_mask
    return need * horizon * jnp.sum(owned_castles * garrison * safety)


def castle_land_potential(
    state: GameState,
    *,
    truncation: int = 1200,
    terminal: jax.Array | bool = False,
) -> jax.Array:
    """Return the zero-sum potential ``[Phi_0(s), Phi_1(s)]``.

    For player zero, with player-one quantities subtracted::

        Phi_0(s) = 0.05 * tanh(land_margin / 20)
                 + 0.05 * tanh(castle_asset_margin / 0.10)
        Phi_1(s) = -Phi_0(s)

    A castle asset is the product of a smooth behind-state ``need`` factor,
    remaining production horizon, garrison quality, and distance from enemy
    land. The total potential is bounded to ``[-0.10, 0.10]`` per player.

    ``terminal`` must include both genuine termination and time-limit
    truncation. It explicitly drains the potential to zero at the final
    transition, which is required for exact telescoping.
    """
    land = jnp.sum(state.ownership & state.board_mask[None], axis=(-2, -1))
    land_margin = land[0].astype(jnp.float32) - land[1].astype(jnp.float32)
    land_growth_events = future_growth_events(
        state.time, period=50, truncation=truncation
    ).astype(jnp.float32)
    land_horizon = jnp.clip(
        land_growth_events / LAND_HORIZON_EVENTS, 0.0, 1.0
    )
    land_term = LAND_POTENTIAL_SCALE * land_horizon * jnp.tanh(
        land_margin / LAND_MARGIN_TEMPERATURE
    )
    castle_zero = _player_castle_asset(state, 0, truncation=truncation)
    castle_one = _player_castle_asset(state, 1, truncation=truncation)
    castle_term = CASTLE_POTENTIAL_SCALE * jnp.tanh(
        (castle_zero - castle_one) / CASTLE_ASSET_TEMPERATURE
    )
    phi_zero = land_term + castle_term
    phi = jnp.stack([phi_zero, -phi_zero])
    is_terminal = jnp.asarray(terminal) | (state.winner >= 0) | (
        state.time >= truncation
    )
    return jnp.where(is_terminal, jnp.zeros_like(phi), phi)


def potential_shaping_reward(
    prior_state: GameState,
    successor_state: GameState,
    *,
    terminated: jax.Array | bool,
    truncated: jax.Array | bool,
    truncation: int = 1200,
    gamma: float = 1.0,
) -> jax.Array:
    """Compute ``gamma * Phi(s') - Phi(s)`` for both players."""
    prior = castle_land_potential(prior_state, truncation=truncation)
    successor = castle_land_potential(
        successor_state,
        truncation=truncation,
        terminal=jnp.asarray(terminated) | jnp.asarray(truncated),
    )
    return gamma * successor - prior

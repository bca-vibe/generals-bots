import jax.numpy as jnp
import numpy as np

from generals.core.game import create_initial_state
from generals.training.potential import (
    CASTLE_POTENTIAL_SCALE,
    LAND_POTENTIAL_SCALE,
    castle_land_potential,
    future_growth_events,
    potential_shaping_reward,
)


def _state():
    grid = jnp.zeros((21, 21), dtype=jnp.int32)
    grid = grid.at[5, 5].set(1).at[15, 15].set(2)
    return create_initial_state(grid)


def test_initial_symmetric_state_has_zero_potential():
    np.testing.assert_allclose(castle_land_potential(_state()), jnp.zeros(2))


def test_growth_horizons_use_exact_live_production_ticks():
    assert int(future_growth_events(jnp.int32(799), period=2, truncation=1200)) == 200
    assert int(future_growth_events(jnp.int32(800), period=2, truncation=1200)) == 199
    assert int(future_growth_events(jnp.int32(799), period=50, truncation=1200)) == 8
    assert int(future_growth_events(jnp.int32(800), period=50, truncation=1200)) == 7
    assert int(future_growth_events(jnp.int32(1199), period=2, truncation=1200)) == 0


def test_potential_is_zero_sum_bounded_and_antisymmetric():
    state = _state()
    ownership = state.ownership.at[0, 5:9, 6:10].set(True)
    ownership = ownership.at[1, 13:16, 12:15].set(True)
    castles = state.castles.at[7, 8].set(True).at[14, 13].set(True)
    armies = state.armies.at[7, 8].set(30).at[14, 13].set(4)
    state = state._replace(ownership=ownership, castles=castles, armies=armies)
    phi = castle_land_potential(state)
    assert float(phi[0]) == -float(phi[1])
    assert abs(float(phi[0])) <= LAND_POTENTIAL_SCALE + CASTLE_POTENTIAL_SCALE

    swapped = state._replace(ownership=state.ownership[::-1])
    swapped_phi = castle_land_potential(swapped)
    np.testing.assert_allclose(swapped_phi, -phi, atol=1e-7)


def test_safe_garrisoned_castle_is_worth_more_than_exposed_empty_castle():
    state = _state()
    ownership = state.ownership.at[0, 5:10, 5:10].set(True)
    ownership = ownership.at[1, 14, 14].set(True)
    safe = state._replace(
        ownership=ownership,
        castles=state.castles.at[7, 7].set(True),
        armies=state.armies.at[7, 7].set(30),
        time=jnp.int32(400),
    )
    exposed_ownership = ownership.at[1, 7, 8].set(True).at[0, 7, 8].set(False)
    exposed = safe._replace(
        ownership=exposed_ownership,
        armies=safe.armies.at[7, 7].set(0),
    )
    assert float(castle_land_potential(safe)[0]) > float(
        castle_land_potential(exposed)[0]
    )


def test_land_value_drains_with_remaining_growth_opportunities():
    state = _state()
    ownership = state.ownership.at[0, 5:9, 5:9].set(True)
    early = state._replace(ownership=ownership, time=jnp.int32(799))
    late = early._replace(time=jnp.int32(1150))
    assert float(castle_land_potential(early)[0]) > float(
        castle_land_potential(late)[0]
    )


def test_terminal_and_truncated_successors_drain_potential():
    state = _state()
    ownership = state.ownership.at[0, 5:9, 5:9].set(True)
    state = state._replace(ownership=ownership)
    phi = castle_land_potential(state)
    terminated = potential_shaping_reward(
        state,
        state._replace(winner=jnp.int32(0)),
        terminated=True,
        truncated=False,
    )
    truncated = potential_shaping_reward(
        state,
        state._replace(time=jnp.int32(1200)),
        terminated=False,
        truncated=True,
    )
    np.testing.assert_allclose(terminated, -phi)
    np.testing.assert_allclose(truncated, -phi)


def test_shaping_telescopes_across_a_trajectory():
    first = _state()
    second = first._replace(ownership=first.ownership.at[0, 5, 6].set(True))
    third = second._replace(
        ownership=second.ownership.at[0, 5, 7].set(True),
        castles=second.castles.at[5, 7].set(True),
        armies=second.armies.at[5, 7].set(20),
    )
    terminal = third._replace(winner=jnp.int32(0))
    rewards = [
        potential_shaping_reward(
            first, second, terminated=False, truncated=False
        ),
        potential_shaping_reward(
            second, third, terminated=False, truncated=False
        ),
        potential_shaping_reward(
            third, terminal, terminated=True, truncated=False
        ),
    ]
    np.testing.assert_allclose(sum(rewards), -castle_land_potential(first), atol=1e-7)

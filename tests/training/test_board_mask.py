import jax.numpy as jnp

from generals.core.game import create_initial_state


def test_padding_is_public_but_not_passable():
    grid = jnp.full((21, 21), -3, dtype=jnp.int32)
    grid = grid.at[:18, :20].set(0)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[17, 19].set(2)

    state = create_initial_state(grid)

    assert int(state.board_mask.sum()) == 18 * 20
    assert bool(state.board_mask[17, 19])
    assert not bool(state.board_mask[18, 0])
    assert not bool(state.passable[18, 0])
    assert bool(state.mountains[18, 0])

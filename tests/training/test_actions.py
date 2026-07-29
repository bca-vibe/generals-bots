import jax.numpy as jnp

from generals.core.observation import Observation
from generals.training.actions import (
    ACTION_COUNT,
    PASS_INDEX,
    decode_action,
    encode_action,
    legal_action_mask,
)


def make_observation(*, armies=None, owned=None, generals=None):
    shape = (21, 21)
    zeros = jnp.zeros(shape, dtype=jnp.bool_)
    armies = jnp.zeros(shape, dtype=jnp.int32) if armies is None else armies
    owned = zeros if owned is None else owned
    generals = zeros if generals is None else generals
    return Observation(
        armies=armies,
        generals=generals,
        castles=zeros,
        mountains=zeros,
        neutral_cells=~owned,
        owned_cells=owned,
        opponent_cells=zeros,
        fog_cells=zeros,
        structures_in_fog=zeros,
        owned_land_count=owned.sum(),
        owned_army_count=(armies * owned).sum(),
        opponent_land_count=jnp.int32(0),
        opponent_army_count=jnp.int32(0),
        timestep=jnp.int32(0),
    )


def test_canonical_action_round_trip():
    actions = [
        jnp.array([0, 4, 7, 3, 0]),
        jnp.array([0, 20, 0, 1, 1]),
        jnp.array([2, 8, 9, 0, 0]),
        jnp.array([1, 0, 0, 0, 0]),
    ]
    assert ACTION_COUNT == 3970
    assert int(encode_action(actions[-1])) == PASS_INDEX
    for action in actions:
        assert jnp.array_equal(decode_action(encode_action(action)), action)


def test_build_mask_uses_exact_observable_price():
    armies = jnp.zeros((21, 21), dtype=jnp.int32).at[10, 11].set(47)
    owned = jnp.zeros((21, 21), dtype=jnp.bool_).at[10, 10].set(True).at[10, 11].set(True)
    generals = jnp.zeros((21, 21), dtype=jnp.bool_).at[10, 10].set(True)
    mask = legal_action_mask(make_observation(armies=armies, owned=owned, generals=generals))
    build_index = 8 * 441 + 10 * 21 + 11
    assert bool(mask[build_index])
    assert bool(mask[PASS_INDEX])

    too_small = armies.at[10, 11].set(46)
    mask = legal_action_mask(make_observation(armies=too_small, owned=owned, generals=generals))
    assert not bool(mask[build_index])


def test_public_board_mask_blocks_moves_into_padding():
    armies = jnp.zeros((21, 21), dtype=jnp.int32).at[17, 19].set(2)
    owned = jnp.zeros((21, 21), dtype=jnp.bool_).at[17, 19].set(True)
    board_mask = jnp.zeros((21, 21), dtype=jnp.bool_).at[:18, :20].set(True)
    mask = legal_action_mask(
        make_observation(armies=armies, owned=owned), board_mask
    )
    source = 17 * 21 + 19
    assert not bool(mask[1 * 441 + source])  # down
    assert not bool(mask[3 * 441 + source])  # right
    assert bool(mask[0 * 441 + source])      # up
    assert bool(mask[2 * 441 + source])      # left

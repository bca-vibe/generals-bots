import jax.numpy as jnp

from generals.core.observation import Observation
from generals.training.observation import augment_observation, init_observation_memory


def make_observation(opponent_cells, armies):
    shape = (21, 21)
    zeros = jnp.zeros(shape, dtype=jnp.bool_)
    return Observation(
        armies=armies,
        generals=zeros,
        castles=zeros,
        mountains=zeros,
        neutral_cells=~opponent_cells,
        owned_cells=zeros,
        opponent_cells=opponent_cells,
        fog_cells=zeros,
        structures_in_fog=zeros,
        owned_land_count=jnp.int32(0),
        owned_army_count=jnp.int32(0),
        opponent_land_count=opponent_cells.sum(),
        opponent_army_count=armies.sum(),
        timestep=jnp.int32(7),
    )


def test_zero_army_enemy_cell_refreshes_last_seen_memory():
    memory = init_observation_memory()
    memory = memory._replace(
        last_seen_enemy_army=memory.last_seen_enemy_army.at[4, 5].set(9),
        last_seen_enemy_age=memory.last_seen_enemy_age.at[4, 5].set(12),
    )
    opponent = jnp.zeros((21, 21), dtype=jnp.bool_).at[4, 5].set(True)
    augmented, memory = augment_observation(
        make_observation(opponent, jnp.zeros((21, 21), dtype=jnp.int32)), memory
    )
    assert augmented.shape == (38, 21, 21)
    assert float(memory.last_seen_enemy_army[4, 5]) == 0.0
    assert float(memory.last_seen_enemy_age[4, 5]) == 0.0


def test_padding_is_encoded_as_known_mountain_not_fog():
    observation = make_observation(
        jnp.zeros((21, 21), dtype=jnp.bool_),
        jnp.zeros((21, 21), dtype=jnp.int32),
    )
    board_mask = jnp.zeros((21, 21), dtype=jnp.bool_).at[:18, :20].set(True)
    augmented, _ = augment_observation(
        observation, init_observation_memory(), board_mask
    )
    assert float(augmented[8, 20, 20]) == 1.0
    assert float(augmented[12, 20, 20]) == 0.0
    assert float(augmented[13, 20, 20]) == 0.0

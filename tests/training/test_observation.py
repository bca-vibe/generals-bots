import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core import game
from generals.core.env import GeneralsEnv
from generals.core.observation import Observation
from generals.modifiers import build_castles
from generals.training.observation import (
    COMPETITION_OBSERVATION_SCHEMA,
    LEGACY_OBSERVATION_SCHEMA,
    augment_observation,
    init_observation_memory,
    normalize_augmented_observation,
    observation_channel_count,
)


def make_observation(opponent_cells=None, armies=None, **overrides):
    shape = (21, 21)
    zeros = jnp.zeros(shape, dtype=jnp.bool_)
    if opponent_cells is None:
        opponent_cells = zeros
    if armies is None:
        armies = jnp.zeros(shape, dtype=jnp.int32)
    defaults = dict(
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
    defaults.update(overrides)
    return Observation(**defaults)


def test_schema_channel_counts():
    assert observation_channel_count(LEGACY_OBSERVATION_SCHEMA, 7) == 38
    assert observation_channel_count(COMPETITION_OBSERVATION_SCHEMA, 7) == 36


def test_zero_army_enemy_cell_refreshes_last_seen_memory():
    memory = init_observation_memory()
    memory = memory._replace(
        last_seen_enemy_army=memory.last_seen_enemy_army.at[4, 5].set(9),
        last_seen_enemy_age=memory.last_seen_enemy_age.at[4, 5].set(12),
    )
    opponent = jnp.zeros((21, 21), dtype=jnp.bool_).at[4, 5].set(True)
    augmented, memory = augment_observation(
        make_observation(opponent, jnp.zeros((21, 21), dtype=jnp.int32)),
        memory,
        observation_schema=LEGACY_OBSERVATION_SCHEMA,
    )
    assert augmented.shape == (38, 21, 21)
    assert float(memory.last_seen_enemy_army[4, 5]) == 0.0
    assert float(memory.last_seen_enemy_age[4, 5]) == 0.0


def test_competition_schema_removes_neutral_army_and_structure_in_fog_channels():
    neutral_armies = jnp.zeros((21, 21), dtype=jnp.int32).at[3, 4].set(17)
    observation = make_observation(armies=neutral_armies)
    legacy, _ = augment_observation(
        observation,
        init_observation_memory(),
        observation_schema=LEGACY_OBSERVATION_SCHEMA,
    )
    competition, _ = augment_observation(
        observation,
        init_observation_memory(),
        observation_schema=COMPETITION_OBSERVATION_SCHEMA,
    )

    assert legacy.shape == (38, 21, 21)
    assert competition.shape == (36, 21, 21)
    assert float(legacy[3, 3, 4]) == 17.0
    assert jnp.array_equal(
        competition, jnp.delete(legacy, jnp.array([3, 13]), axis=0)
    )

    normalized_legacy = normalize_augmented_observation(
        legacy, LEGACY_OBSERVATION_SCHEMA
    )
    normalized_competition = normalize_augmented_observation(
        competition, COMPETITION_OBSERVATION_SCHEMA
    )
    assert jnp.array_equal(
        normalized_competition,
        jnp.delete(normalized_legacy, jnp.array([3, 13]), axis=0),
    )


def test_padding_is_encoded_as_known_mountain_not_fog():
    observation = make_observation()
    board_mask = jnp.zeros((21, 21), dtype=jnp.bool_).at[:18, :20].set(True)
    augmented, _ = augment_observation(
        observation,
        init_observation_memory(),
        board_mask,
        LEGACY_OBSERVATION_SCHEMA,
    )
    assert float(augmented[8, 20, 20]) == 1.0
    assert float(augmented[12, 20, 20]) == 0.0
    assert float(augmented[13, 20, 20]) == 0.0


def _structure_observation(target, *, fog=False, structure=False, castle=False, mountain=False):
    shape = (21, 21)
    zeros = jnp.zeros(shape, dtype=jnp.bool_)
    fog_cells = zeros.at[target].set(fog)
    structures = zeros.at[target].set(structure)
    castles = zeros.at[target].set(castle)
    mountains = zeros.at[target].set(mountain)
    neutral = ~(fog_cells | structures | castles | mountains)
    return make_observation(
        neutral_cells=neutral,
        fog_cells=fog_cells,
        structures_in_fog=structures,
        castles=castles,
        mountains=mountains,
    )


def test_visible_plain_then_fogged_structure_is_inferred_as_castle():
    target = (8, 9)
    memory = init_observation_memory()
    _, memory = augment_observation(
        _structure_observation(target),
        memory,
        observation_schema=COMPETITION_OBSERVATION_SCHEMA,
    )
    augmented, memory = augment_observation(
        _structure_observation(target, structure=True),
        memory,
        observation_schema=COMPETITION_OBSERVATION_SCHEMA,
    )
    assert bool(memory.known_castles[target])
    assert float(augmented[6][target]) == 1.0


def test_ordinary_fog_counts_as_plain_evidence_for_castle_inference():
    target = (8, 9)
    memory = init_observation_memory()
    _, memory = augment_observation(
        _structure_observation(target, fog=True),
        memory,
        observation_schema=COMPETITION_OBSERVATION_SCHEMA,
    )
    assert bool(memory.ever_plain[target])
    augmented, _ = augment_observation(
        _structure_observation(target, structure=True),
        memory,
        observation_schema=COMPETITION_OBSERVATION_SCHEMA,
    )
    assert float(augmented[6][target]) == 1.0


def test_starting_fogged_structure_is_recorded_as_known_mountain():
    target = (8, 9)
    augmented, memory = augment_observation(
        _structure_observation(target, structure=True),
        init_observation_memory(),
        observation_schema=COMPETITION_OBSERVATION_SCHEMA,
    )
    assert not bool(memory.known_castles[target])
    assert float(augmented[6][target]) == 0.0
    assert bool(memory.known_mountains[target])
    assert float(augmented[7][target]) == 1.0


def test_known_mountain_is_never_inferred_as_castle():
    target = (8, 9)
    _, memory = augment_observation(
        _structure_observation(target, mountain=True),
        init_observation_memory(),
        observation_schema=COMPETITION_OBSERVATION_SCHEMA,
    )
    augmented, memory = augment_observation(
        _structure_observation(target, structure=True),
        memory,
        observation_schema=COMPETITION_OBSERVATION_SCHEMA,
    )
    assert bool(memory.known_mountains[target])
    assert not bool(memory.known_castles[target])
    assert float(augmented[6][target]) == 0.0


def test_visible_and_previously_known_castles_persist_in_both_schemas():
    target = (8, 9)
    for schema, castle_channel in (
        (LEGACY_OBSERVATION_SCHEMA, 7),
        (COMPETITION_OBSERVATION_SCHEMA, 6),
    ):
        _, memory = augment_observation(
            _structure_observation(target, castle=True),
            init_observation_memory(),
            observation_schema=schema,
        )
        augmented, memory = augment_observation(
            _structure_observation(target, structure=True),
            memory,
            observation_schema=schema,
        )
        assert bool(memory.known_castles[target])
        assert float(augmented[castle_channel][target]) == 1.0


def test_legacy_schema_does_not_infer_a_built_castle():
    target = (8, 9)
    _, memory = augment_observation(
        _structure_observation(target, fog=True),
        init_observation_memory(),
        observation_schema=LEGACY_OBSERVATION_SCHEMA,
    )
    augmented, memory = augment_observation(
        _structure_observation(target, structure=True),
        memory,
        observation_schema=LEGACY_OBSERVATION_SCHEMA,
    )
    assert not bool(memory.known_castles[target])
    assert float(augmented[7][target]) == 0.0


def test_competition_build_transitions_fog_to_structure_on_same_turn():
    environment = GeneralsEnv(mode="competition", pool_size=1)
    state = environment._make_single_state_fixed(jrandom.PRNGKey(20260730), 21, 21)
    p0_visibility = game.get_visibility(state.ownership[0])

    candidates = (
        state.board_mask
        & ~state.mountains
        & ~state.generals
        & ~state.castles
        & ~p0_visibility
    )
    candidate_indices = jnp.argwhere(candidates, size=21 * 21, fill_value=-1)
    target_array = candidate_indices[0]
    assert bool(jnp.all(target_array >= 0))
    target = (int(target_array[0]), int(target_array[1]))

    state = state._replace(
        armies=state.armies.at[target].set(100),
        ownership=state.ownership.at[0, target[0], target[1]].set(False)
        .at[1, target[0], target[1]].set(True),
        ownership_neutral=state.ownership_neutral.at[target].set(False),
    )
    before = game.get_observation(state, 0)
    assert bool(before.fog_cells[target])
    assert not bool(before.structures_in_fog[target])
    assert not bool(before.castles[target])

    competition_memory = init_observation_memory()
    _, competition_memory = augment_observation(
        before,
        competition_memory,
        state.board_mask,
        COMPETITION_OBSERVATION_SCHEMA,
    )
    legacy_memory = init_observation_memory()
    _, legacy_memory = augment_observation(
        before,
        legacy_memory,
        state.board_mask,
        LEGACY_OBSERVATION_SCHEMA,
    )

    actions = jnp.stack(
        [
            jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32),
            jnp.array(
                [build_castles.BUILD, target[0], target[1], 0, 0],
                dtype=jnp.int32,
            ),
        ]
    )
    pool = jax.tree.map(lambda value: value[None], state)
    timestep, new_state = environment.step(state, actions, pool)
    assert bool(new_state.castles[target])

    after = jax.tree.map(lambda value: value[0], timestep.observation)
    assert not bool(after.fog_cells[target])
    assert bool(after.structures_in_fog[target])
    assert not bool(after.castles[target])

    competition, competition_memory = augment_observation(
        after,
        competition_memory,
        new_state.board_mask,
        COMPETITION_OBSERVATION_SCHEMA,
    )
    legacy, legacy_memory = augment_observation(
        after,
        legacy_memory,
        new_state.board_mask,
        LEGACY_OBSERVATION_SCHEMA,
    )
    assert bool(competition_memory.known_castles[target])
    assert float(competition[6][target]) == 1.0
    assert not bool(legacy_memory.known_castles[target])
    assert float(legacy[7][target]) == 0.0

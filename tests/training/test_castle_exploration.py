import jax
import jax.numpy as jnp
import numpy as np

from generals.core.game import create_initial_state, get_observation
from generals.training.actions import CELL_COUNT, MOVE_PLANES, PASS_INDEX
from generals.training.castle_exploration import (
    action_distribution_statistics,
    apply_tactical_build_logit_boost,
    tactical_build_mask,
)
from generals.training.observation import augment_observation, init_observation_memory


def _eligible_state():
    grid = jnp.zeros((21, 21), dtype=jnp.int32)
    grid = grid.at[5, 5].set(1).at[15, 15].set(2)
    state = create_initial_state(grid)
    own_site = (5, 12)
    enemy_cells = [(14, 14), (14, 15), (14, 16), (15, 14), (15, 16), (16, 14)]
    ownership = state.ownership.at[0, own_site[0], own_site[1]].set(True)
    ownership_neutral = state.ownership_neutral.at[own_site].set(False)
    for row, column in enemy_cells:
        ownership = ownership.at[1, row, column].set(True)
        ownership_neutral = ownership_neutral.at[row, column].set(False)
    armies = state.armies.at[own_site].set(50)
    return state._replace(
        ownership=ownership,
        ownership_neutral=ownership_neutral,
        armies=armies,
        time=jnp.int32(100),
    ), own_site


def _mask_for_state(state):
    observation = get_observation(state, 0)
    _, memory = augment_observation(
        observation,
        init_observation_memory(),
        state.board_mask,
        "competition_39",
        800,
    )
    return tactical_build_mask(observation, memory, state.board_mask), memory


def test_tactical_gate_checks_live_price_reserve_and_payback():
    state, site = _eligible_state()
    mask, _ = _mask_for_state(state)
    assert bool(mask[site])

    late_mask, _ = _mask_for_state(state._replace(time=jnp.int32(1100)))
    assert not bool(late_mask[site])

    underfunded = state._replace(armies=state.armies.at[site].set(40))
    underfunded_mask, _ = _mask_for_state(underfunded)
    assert not bool(underfunded_mask[site])


def test_tactical_gate_rejects_recently_remembered_enemy_land():
    state, site = _eligible_state()
    observation = get_observation(state, 0)
    _, memory = augment_observation(
        observation,
        init_observation_memory(),
        state.board_mask,
        "competition_39",
        800,
    )
    remembered = (site[0], site[1] + 2)
    memory = memory._replace(
        last_seen_enemy_owned=memory.last_seen_enemy_owned.at[remembered].set(True),
        last_seen_enemy_age=memory.last_seen_enemy_age.at[remembered].set(10),
    )
    mask = tactical_build_mask(observation, memory, state.board_mask)
    assert not bool(mask[site])


def test_fixed_logit_boost_is_normalized_recomputable_and_differentiable():
    build_start = MOVE_PLANES * CELL_COUNT
    logits = jnp.full((PASS_INDEX + 1,), -1e9)
    logits = logits.at[0].set(0.0).at[build_start + 7].set(-10.0)
    logits = logits.at[PASS_INDEX].set(-2.0)
    legal = logits > -1e8
    eligible = (
        jnp.zeros((CELL_COUNT,), dtype=jnp.bool_).at[7].set(True).reshape(21, 21)
    )
    boosted = apply_tactical_build_logit_boost(logits, eligible, 10.0)
    probabilities = jax.nn.softmax(boosted)
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-7)

    first = jax.nn.log_softmax(boosted)[build_start + 7]
    second = jax.nn.log_softmax(
        apply_tactical_build_logit_boost(logits, eligible, 10.0)
    )[build_start + 7]
    np.testing.assert_allclose(first, second, atol=0.0)

    def selected_log_probability(build_logit):
        candidate = logits.at[build_start + 7].set(build_logit)
        behavior = apply_tactical_build_logit_boost(candidate, eligible, 10.0)
        return jax.nn.log_softmax(behavior)[build_start + 7]

    assert float(jax.grad(selected_log_probability)(-10.0)) > 0.0
    statistics = action_distribution_statistics(boosted, legal)
    assert float(statistics["build_probability"]) > 0.4

import jax.numpy as jnp

from generals.core.observation import Observation
from generals.training.actions import ACTION_COUNT, CELL_COUNT, MOVE_PLANES
from generals.training.evaluation import _empty_behavior, _update_behavior_before_step
from generals.training.train import _behavior_rates


def _observation():
    spatial_bool = jnp.zeros((1, 21, 21), dtype=jnp.bool_)
    spatial_int = jnp.zeros((1, 21, 21), dtype=jnp.int32)
    return Observation(
        armies=spatial_int,
        generals=spatial_bool,
        castles=spatial_bool,
        mountains=spatial_bool,
        neutral_cells=spatial_bool.at[0, 1, 2].set(True),
        owned_cells=spatial_bool.at[0, 1, 1].set(True),
        opponent_cells=spatial_bool.at[0, 2, 1].set(True),
        fog_cells=spatial_bool,
        structures_in_fog=spatial_bool,
        owned_land_count=jnp.array([1]),
        owned_army_count=jnp.array([10]),
        opponent_land_count=jnp.array([1]),
        opponent_army_count=jnp.array([10]),
        timestep=jnp.array([0]),
    )


def test_behavior_counts_build_opportunities_and_immediate_move_reversals():
    observation = _observation()
    legal = jnp.zeros((1, ACTION_COUNT), dtype=jnp.bool_)
    legal = legal.at[0, MOVE_PLANES * CELL_COUNT + 22].set(True)
    active = jnp.array([True])
    behavior = _empty_behavior(1)
    behavior = _update_behavior_before_step(
        behavior,
        observation,
        legal,
        jnp.array([[0, 1, 1, 3, 1]]),
        active,
    )
    behavior = _update_behavior_before_step(
        behavior,
        observation,
        legal,
        jnp.array([[0, 1, 2, 2, 0]]),
        active,
    )
    behavior = _update_behavior_before_step(
        behavior,
        observation,
        legal,
        jnp.array([[2, 1, 1, 0, 0]]),
        active,
    )

    assert int(behavior["actions"]) == 3
    assert int(behavior["moves"]) == 2
    assert int(behavior["half_moves"]) == 1
    assert int(behavior["expansion_moves"]) == 1
    assert int(behavior["reinforce_moves"]) == 1
    assert int(behavior["dithers"]) == 1
    assert int(behavior["moves_after_move"]) == 1
    assert int(behavior["builds"]) == 1
    assert int(behavior["build_opportunity_steps"]) == 3


def test_behavior_rate_names_exclude_fog_and_general_source_metrics():
    rates = _behavior_rates(
        {
            "behavior_actions": 10,
            "behavior_moves": 8,
            "behavior_builds": 2,
            "behavior_build_opportunity_steps": 4,
            "behavior_games_with_build": 1,
            "behavior_games_with_build_opportunity": 2,
            "behavior_completed_games": 2,
            "behavior_dithers": 1,
            "behavior_moves_after_move": 4,
        },
        "behavior_",
    )
    assert rates["behavior/castle_build/legal_step_rate"] == 0.5
    assert rates["behavior/castle_build/legal_game_rate"] == 0.5
    assert rates["behavior/dither/move_rate"] == 0.125
    assert not any("fog" in name or "general_source" in name for name in rates)

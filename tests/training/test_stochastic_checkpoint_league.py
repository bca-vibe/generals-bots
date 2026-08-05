import jax
import jax.numpy as jnp
import pytest

from generals.training.config import TrainingConfig
from generals.training.evaluation import (
    _empty_behavior,
    _record_successful_builds,
    evaluate_paired_networks,
    round_robin_matrices,
)
from generals.training.train import build_network, make_environment


def _tiny_config(**overrides):
    values = dict(
        run_name="stochastic_league_test",
        observation_schema="competition_39",
        model_architecture="transformer",
        temporal_window=32,
        depth=1,
        embed_dim=32,
        attention_heads=4,
        ff_factor=2,
        use_bf16=False,
        value_bins=16,
        num_envs=2,
        num_steps=8,
        minibatch_size=4,
        pool_size=16,
        truncation=4,
    )
    values.update(overrides)
    return TrainingConfig(**values)


def test_successful_build_accounting_tracks_creation_and_game_flag():
    behavior = _empty_behavior(2)
    before = jnp.zeros((2, 4, 4), dtype=jnp.bool_)
    after = before.at[0, 1, 2].set(True)
    actions = jnp.array([[2, 1, 2, 0, 0], [2, 2, 3, 0, 0]])
    result = _record_successful_builds(behavior, before, after, actions, jnp.array([True, True]))
    assert int(result["successful_builds"]) == 1
    assert result["had_successful_build"].tolist() == [True, False]


def test_matrix_payload_attributes_rows_and_averages_self_play():
    matches = [
        {
            "a": "alpha",
            "b": "alpha",
            "score": 0.5,
            "win_rate": 0.4,
            "games": 10,
            "losses": 4,
            "behavior_a_successful_builds": 2,
            "behavior_b_successful_builds": 4,
            "behavior_a_games_with_successful_build": 1,
            "behavior_b_games_with_successful_build": 3,
        },
        {
            "a": "alpha",
            "b": "beta",
            "score": 0.7,
            "win_rate": 0.6,
            "games": 10,
            "losses": 2,
            "behavior_a_successful_builds": 5,
            "behavior_b_successful_builds": 7,
            "behavior_a_games_with_successful_build": 4,
            "behavior_b_games_with_successful_build": 6,
        },
        {
            "a": "beta",
            "b": "beta",
            "score": 0.5,
            "win_rate": 0.3,
            "games": 10,
            "losses": 3,
            "behavior_a_successful_builds": 8,
            "behavior_b_successful_builds": 10,
            "behavior_a_games_with_successful_build": 5,
            "behavior_b_games_with_successful_build": 7,
        },
    ]
    matrices = round_robin_matrices(["alpha", "beta"], matches)
    assert matrices["score"][0] == [0.5, 0.7]
    assert matrices["score"][1] == pytest.approx([0.3, 0.5])
    assert matrices["win_rate"] == [[0.4, 0.6], [0.2, 0.3]]
    assert matrices["castles_built"] == [[3.0, 5], [7, 9.0]]
    assert matrices["games_with_castle"] == [[2.0, 4], [6, 6.0]]


def test_categorical_paired_evaluator_is_reproducible():
    config = _tiny_config()
    environment = make_environment(config, config.curriculum[0], pool_size=16)
    key = jax.random.PRNGKey(17)
    pool, _ = environment.reset(key)
    network = build_network(config, key)

    def run():
        return evaluate_paired_networks(
            environment,
            pool,
            network,
            network,
            2,
            config.truncation,
            schema_a=config.observation_schema,
            schema_b=config.observation_schema,
            pad_to=config.pad_to,
            history_size=config.history_size,
            temporal_window=config.temporal_window,
            sampling="categorical",
            key=jax.random.PRNGKey(99),
        )

    first = run()
    second = run()
    assert all(float(first[name]) == float(second[name]) for name in first)
    assert int(first["wins"] + first["losses"] + first["draws"]) == 4

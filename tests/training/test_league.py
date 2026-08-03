"""Post-training league configuration and accounting tests."""

from pathlib import Path

import jax

from generals.training.config import TrainingConfig
from generals.training.league import (
    aggregate_league_results,
    make_opponent_action,
    make_opponent_policy,
)
from generals.training.train import _make_opponent_evaluator, build_network, make_environment, train

CONFIG = (
    Path(__file__).resolve().parents[2]
    / "generals/training/configs/smoke_conv_d384_rms025_1xh100_50iter.toml"
)


def tiny_config(**overrides):
    values = dict(
        run_name="league_test",
        observation_schema="competition_39",
        model_architecture="conv_transformer",
        temporal_window=32,
        depth=1,
        embed_dim=32,
        attention_heads=4,
        ff_factor=2,
        use_bf16=False,
        value_bins=16,
        conv_channels=12,
        conv_groups=3,
        num_envs=2,
        num_steps=8,
        minibatch_size=4,
        pool_size=16,
        eval_games=4,
        truncation=10,
    )
    values.update(overrides)
    return TrainingConfig(**values)


def test_smoke_recipe_locks_requested_architecture_and_league():
    config = TrainingConfig.from_toml(CONFIG)
    assert config.embed_dim == 384
    assert config.conv_initial_token_rms_ratio == 0.25
    assert config.num_iterations == 50
    assert config.league_eval_after_training
    assert len(config.league_opponents) == 12
    assert config.league_eval_maps == 256


def test_aggregate_reports_pooled_and_macro_scores():
    aggregate = aggregate_league_results(
        {
            "alpha": {"wins": 3.0, "losses": 1.0, "draws": 0.0, "score": 0.75},
            "beta": {"wins": 0.0, "losses": 2.0, "draws": 2.0, "score": 0.25},
        }
    )
    assert aggregate == {
        "wins": 3.0,
        "losses": 3.0,
        "draws": 2.0,
        "games": 8.0,
        "score": 0.5,
        "macro_score": 0.5,
    }


def test_generalized_evaluator_is_deterministic_and_swaps_both_seats():
    config = tiny_config()
    environment = make_environment(config, config.curriculum[0], pool_size=16)
    key = jax.random.PRNGKey(9)
    pool, _ = environment.reset(key)
    network = build_network(config, key)
    evaluator = _make_opponent_evaluator(
        config, environment, n_maps=2, truncation=config.truncation, opponent_name="random"
    )
    first, _ = evaluator(pool, network, key)
    second, _ = evaluator(pool, network, key)
    assert int(first["wins"] + first["losses"] + first["draws"]) == 4
    assert all(float(first[name]) == float(second[name]) for name in first)


def test_human_exe_policy_exposes_functional_match_memory():
    policy = make_opponent_policy("human_exe")
    memory = policy.initial_memory(21)
    assert memory.enemy_origin_score.shape == (21, 21)


def test_unknown_opponent_and_incomplete_trace_controls_fail_fast(tmp_path):
    try:
        make_opponent_action("missing")
    except ValueError as error:
        assert "Unknown league opponent" in str(error)
    else:
        raise AssertionError("unknown opponent was accepted")

    config = tiny_config(output_dir=str(tmp_path), num_iterations=1)
    try:
        train(config, trace_iterations=1)
    except ValueError as error:
        assert "trace_dir is required" in str(error)
    else:
        raise AssertionError("trace window without an output directory was accepted")

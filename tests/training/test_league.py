"""Post-training league configuration and accounting tests."""

import json
from pathlib import Path

import equinox as eqx
import jax
import pytest

import generals.training.train as train_module
from generals.training.config import TrainingConfig
from generals.training.league import (
    aggregate_league_results,
    make_opponent_action,
    make_opponent_policy,
)
from generals.training.tracking import WandbTracker
from generals.training.train import (
    _combine_sharded_evaluation,
    _first_pmap_replica_to_host,
    _make_opponent_evaluator,
    _replicate_for_pmap,
    _run_league,
    build_network,
    make_environment,
    train,
)

CONFIG = (
    Path(__file__).resolve().parents[2]
    / "generals/training/configs/smoke_conv_d384_rms025_1xh100_50iter.toml"
)
AB_CONFIG_DIR = CONFIG.parents[0]
CONTINUATION_CONFIG = AB_CONFIG_DIR / "conv_d448_8xh100_12h_cont_1313.toml"


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


def test_sharded_evaluation_combines_counts_and_paired_dispersion():
    combined = _combine_sharded_evaluation(
        {
            "wins": [3, 1],
            "losses": [1, 1],
            "draws": [0, 2],
            "score": [0.75, 0.5],
            "paired_score_std": [0.25, 0.0],
        }
    )
    assert combined["games"] == 8
    assert combined["score"] == 0.625
    assert combined["paired_score_std"] > 0


def test_ab_recipes_lock_independent_milestones_and_six_opponent_league():
    transformer = TrainingConfig.from_toml(
        AB_CONFIG_DIR / "arch_ab_d448_8xh100_5h_transformer.toml"
    )
    conv = TrainingConfig.from_toml(
        AB_CONFIG_DIR / "arch_ab_d448_8xh100_5h_conv.toml"
    )
    expected_opponents = (
        "boss",
        "random",
        "hunter",
        "harvester",
        "raider",
        "deathtouch_clock",
    )
    for config in (transformer, conv):
        assert config.seed == 44
        assert config.embed_dim == 448
        assert config.eval_every == 50
        assert config.latest_checkpoint_every == 50
        assert config.checkpoint_every == 200
        assert config.league_eval_every == 200
        assert config.league_eval_maps == 256
        assert config.league_opponents == expected_opponents
        assert config.league_eval_policies == ("raw", "ema")
    assert transformer.model_architecture == "transformer"
    assert conv.model_architecture == "conv_transformer"
    assert conv.conv_initial_token_rms_ratio == 0.25


def test_continuation_recipe_preserves_global_batches_and_lineage():
    config = TrainingConfig.from_toml(CONTINUATION_CONFIG)
    assert config.wandb_run_id == "conv-d448-12h-cont-from-001313-20260803"
    assert config.wandb_job_type == "training-continuation"
    assert config.parent_final_iteration == 1313
    assert config.parent_final_samples == 2_753_560_576
    assert config.resume_raw_weights
    assert config.resume_optimizer_state
    assert config.resume_ema_weights
    assert config.num_envs == 256
    assert config.minibatch_size == 512
    assert 8 * config.num_envs == config.preserved_global_envs
    assert 8 * config.minibatch_size == config.preserved_global_minibatch_size
    assert config.eval_every == 0
    assert config.checkpoint_every == 500
    assert config.league_eval_every == 400
    assert config.league_eval_policies == ("ema",)
    assert config.league_checkpoint_policy == "ema"
    assert config.league_checkpoint_maps == 128


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


def test_sharded_league_writes_policy_namespaces_and_exact_game_count(tmp_path):
    league_maps = max(2, jax.device_count())
    config = tiny_config(
        output_dir=str(tmp_path),
        model_architecture="transformer",
        league_eval_maps=league_maps,
        league_opponents=("random",),
        league_eval_policies=("raw",),
    )
    network = build_network(config, jax.random.PRNGKey(18))
    parameters, static = eqx.partition(network, eqx.is_inexact_array)
    parameters = _replicate_for_pmap(parameters)
    network = eqx.combine(_first_pmap_replica_to_host(parameters), static)
    payload = _run_league(
        config,
        {"raw": network},
        WandbTracker(),
        tmp_path,
        200,
        label="000200",
    )
    result = payload["policies"]["raw"]["opponents"]["random"]
    assert result["games"] == 2 * league_maps
    assert (tmp_path / "league_000200.json").is_file()


def test_human_exe_policy_exposes_functional_match_memory():
    policy = make_opponent_policy("human_exe")
    memory = policy.initial_memory(21)
    assert memory.enemy_origin_score.shape == (21, 21)


def test_milestone_checkpoint_precedes_periodic_league(tmp_path, monkeypatch):
    config = tiny_config(
        output_dir=str(tmp_path),
        run_name="checkpoint_before_league",
        model_architecture="transformer",
        use_bf16=True,
        num_iterations=1,
        eval_every=0,
        latest_checkpoint_every=1,
        checkpoint_every=1,
        metrics_every=1,
        league_eval_after_training=False,
        league_eval_every=1,
        league_eval_maps=max(2, jax.device_count()),
        league_opponents=("random",),
        league_eval_policies=("ema",),
        reset_pool_every=0,
    )

    def fail_league(*_args, **_kwargs):
        raise RuntimeError("synthetic league failure")

    monkeypatch.setattr(train_module, "_run_league", fail_league)
    with pytest.raises(RuntimeError, match="synthetic league failure"):
        train_module.train(config)

    run_dir = config.run_dir
    assert (run_dir / "checkpoint_000001.eqx").is_file()
    metadata = json.loads(
        (run_dir / "latest_checkpoint.json").read_text(encoding="utf-8")
    )
    assert metadata["iteration"] == 1
    assert metadata["archive"] is True


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

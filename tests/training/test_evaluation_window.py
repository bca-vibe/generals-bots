"""Regression: evaluation must respect a non-default temporal_window."""

import jax
import pytest

from generals.training.config import TrainingConfig
from generals.training.observation import (
    COMPETITION_OBSERVATION_SCHEMA,
    LEGACY_OBSERVATION_SCHEMA,
)
from generals.training.train import _make_evaluator, build_network, make_environment


@pytest.mark.parametrize(
    ("observation_schema", "model_architecture"),
    [
        (LEGACY_OBSERVATION_SCHEMA, "transformer"),
        (COMPETITION_OBSERVATION_SCHEMA, "transformer"),
        (COMPETITION_OBSERVATION_SCHEMA, "conv_transformer"),
    ],
)
def test_evaluator_with_non_default_temporal_window(
    observation_schema, model_architecture
):
    config = TrainingConfig(
        observation_schema=observation_schema,
        model_architecture=model_architecture,
        temporal_window=64,
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
        pool_size=4,
        eval_games=4,
        truncation=10,
    )
    stage = config.curriculum[0]
    # pool sizes below 16 silently produce an empty pool (generator batching);
    # 16 is the same floor train.py applies to its eval pool.
    environment = make_environment(config, stage, pool_size=16)
    key = jax.random.PRNGKey(0)
    pool, _ = environment.reset(key)
    network = build_network(config, key)
    evaluator = _make_evaluator(config, environment, 2, config.truncation)
    evaluation, _ = evaluator(pool, network, key)
    total = int(evaluation["wins"]) + int(evaluation["losses"]) + int(evaluation["draws"])
    assert total == 4  # 2 maps x 2 seats, every game accounted for

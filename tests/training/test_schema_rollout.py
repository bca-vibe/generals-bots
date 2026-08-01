import jax
import pytest

from generals.core.env import GeneralsEnv
from generals.training.config import TrainingConfig
from generals.training.observation import (
    COMPETITION_OBSERVATION_SCHEMA,
    LEGACY_OBSERVATION_SCHEMA,
    init_observation_memory,
)
from generals.training.rollout import collect_self_play_rollout
from generals.training.train import build_network


@pytest.mark.parametrize(
    ("observation_schema", "model_architecture", "input_channels"),
    [
        (LEGACY_OBSERVATION_SCHEMA, "transformer", 38),
        (COMPETITION_OBSERVATION_SCHEMA, "transformer", 36),
        (COMPETITION_OBSERVATION_SCHEMA, "conv_transformer", 36),
    ],
)
def test_rollout_uses_selected_observation_schema(
    observation_schema, model_architecture, input_channels
):
    environment = GeneralsEnv(
        grid_dims=(21, 21),
        pad_to=21,
        pool_size=2,
        mountain_density_range=(0.1, 0.1),
        num_castles_range=(0, 0),
        build_castles=True,
    )
    pool, _ = environment.reset(jax.random.PRNGKey(0))
    states = jax.tree.map(lambda value: value[:1], pool)
    memory = init_observation_memory()
    memory = jax.tree.map(lambda value: value[None], memory)
    config = TrainingConfig(
        observation_schema=observation_schema,
        model_architecture=model_architecture,
        depth=1,
        embed_dim=32,
        attention_heads=4,
        ff_factor=2,
        value_bins=16,
        use_bf16=False,
        conv_channels=12,
        conv_groups=3,
    )
    network = build_network(config, jax.random.PRNGKey(1))

    _, rollout, _, _, _ = collect_self_play_rollout(
        states,
        pool,
        environment,
        network,
        jax.random.PRNGKey(2),
        memory,
        memory,
        1,
        observation_schema=observation_schema,
    )
    observations = rollout[0]
    assert observations.shape == (1, 2, input_channels, 21, 21)

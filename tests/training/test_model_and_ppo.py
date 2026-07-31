import jax
import jax.numpy as jnp
import pytest

from generals.training.model import CompetitionTransformer
from generals.training.observation import (
    COMPETITION_OBSERVATION_SCHEMA,
    LEGACY_OBSERVATION_SCHEMA,
)
from generals.training.ppo import compute_gae


@pytest.mark.parametrize(
    ("observation_schema", "input_channels"),
    [
        (LEGACY_OBSERVATION_SCHEMA, 38),
        (COMPETITION_OBSERVATION_SCHEMA, 37),
    ],
)
def test_model_shapes_and_canonical_pass(observation_schema, input_channels):
    model = CompetitionTransformer(
        input_channels=input_channels,
        depth=1,
        model_dim=32,
        heads=4,
        ff_factor=2,
        value_bins=16,
        use_bf16=False,
        observation_schema=observation_schema,
        key=jax.random.PRNGKey(0),
    )
    assert model.patch_embedding.in_features == input_channels * 3 * 3
    observation = jnp.zeros((input_channels, 21, 21), dtype=jnp.float32)
    history = jnp.zeros((2, 512), dtype=jnp.float32)
    mask = jnp.zeros((3970,), dtype=jnp.bool_).at[-1].set(True)
    action_index, action, value, log_probability, entropy, value_logits = model(
        observation, history, mask, jax.random.PRNGKey(1)
    )
    assert int(action_index) == 3969
    assert jnp.array_equal(action, jnp.array([1, 0, 0, 0, 0]))
    assert value.shape == ()
    assert log_probability.shape == ()
    assert entropy.shape == ()
    assert value_logits.shape == (16,)


def test_gae_treats_hard_draw_as_terminal():
    rewards = jnp.array([[0.0], [0.0]])
    values = jnp.array([[0.4], [0.3]])
    next_values = jnp.array([[0.3], [0.9]])
    terminated = jnp.array([[False], [False]])
    truncated = jnp.array([[False], [True]])
    advantages = compute_gae(
        rewards, values, next_values, terminated, truncated, gamma=1.0, gae_lambda=0.9
    )
    assert jnp.allclose(advantages[-1], -0.3)

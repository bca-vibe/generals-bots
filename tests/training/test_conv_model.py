from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from generals.core.env import GeneralsEnv
from generals.training.config import TrainingConfig
from generals.training.conv_model import (
    ConvCompetitionTransformer,
    ConvPatchResidual,
    calibrate_conv_token_rms,
)
from generals.training.model import CompetitionTransformer
from generals.training.train import (
    _conv_calibration_observations,
    _learning_rate,
    _load_checkpoint_state,
    _save_checkpoint,
    build_network,
)

CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "generals" / "training" / "configs"
)


def tiny_config(architecture: str) -> TrainingConfig:
    return TrainingConfig(
        observation_schema="competition_39",
        model_architecture=architecture,
        depth=1,
        embed_dim=32,
        attention_heads=4,
        ff_factor=2,
        use_bf16=False,
        value_bins=16,
        conv_channels=12,
        conv_groups=3,
        num_envs=4,
        num_steps=8,
        minibatch_size=4,
    )


def optimizer_for(config, network):
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(partial(_learning_rate, config)),
    )
    return optimizer, optimizer.init(eqx.filter(network, eqx.is_inexact_array))


def test_conv_patch_residual_shape_and_nonzero_initial_output():
    stem = ConvPatchResidual(
        input_channels=39,
        conv_channels=96,
        model_dim=448,
        groups=12,
        board_size=21,
        patch_size=3,
        key=jax.random.PRNGKey(0),
    )
    correction = stem(jax.random.normal(jax.random.PRNGKey(1), (39, 21, 21)))
    assert correction.shape == (49, 448)
    assert float(jnp.max(jnp.abs(correction))) > 0.0


def test_data_dependent_calibration_hits_ten_percent_token_rms():
    network = build_network(tiny_config("conv_transformer"), jax.random.PRNGKey(30))
    observations = jax.random.normal(
        jax.random.PRNGKey(31), (16, 39, 21, 21)
    )
    calibrated, metrics = calibrate_conv_token_rms(network, observations, 0.10)
    jax.block_until_ready(metrics)

    assert float(metrics["ratio_before"]) > 0.0
    assert float(metrics["projection_multiplier"]) > 0.0
    assert float(metrics["ratio_after"]) == pytest.approx(0.10, abs=1e-5)

    # Calibration changes only the convolutional output projection. The matched
    # transformer backbone remains bit-identical to its pre-calibration state.
    for before, after in zip(
        jax.tree.leaves(eqx.filter(network.transformer, eqx.is_inexact_array)),
        jax.tree.leaves(eqx.filter(calibrated.transformer, eqx.is_inexact_array)),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)


def test_calibration_batch_uses_real_two_seat_augmented_observations():
    config = replace(
        tiny_config("conv_transformer"), conv_calibration_samples=8
    )
    environment = GeneralsEnv(
        grid_dims=(21, 21),
        pad_to=21,
        pool_size=4,
        build_castles=True,
        deathtouch_turn=800,
    )
    pool, _ = environment.reset(jax.random.PRNGKey(32))
    observations = _conv_calibration_observations(config, pool)

    assert observations.shape == (8, 39, 21, 21)
    network = build_network(config, jax.random.PRNGKey(33))
    _, metrics = calibrate_conv_token_rms(network, observations, 0.10)
    assert float(metrics["ratio_after"]) == pytest.approx(0.10, abs=1e-5)


def test_matched_seed_preserves_identical_transformer_parameters():
    key = jax.random.PRNGKey(2)
    pure = build_network(tiny_config("transformer"), key)
    convolutional = build_network(tiny_config("conv_transformer"), key)
    assert isinstance(pure, CompetitionTransformer)
    assert isinstance(convolutional, ConvCompetitionTransformer)
    for pure_leaf, conv_leaf in zip(
        jax.tree.leaves(eqx.filter(pure, eqx.is_inexact_array)),
        jax.tree.leaves(
            eqx.filter(convolutional.transformer, eqx.is_inexact_array)
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(pure_leaf, conv_leaf)


def test_small_random_projection_gives_every_conv_layer_immediate_gradients():
    network = build_network(tiny_config("conv_transformer"), jax.random.PRNGKey(20))
    observation = jax.random.normal(jax.random.PRNGKey(21), (39, 21, 21))
    history = jnp.zeros((2, 512), dtype=jnp.float32)
    mask = jnp.ones((3970,), dtype=jnp.bool_)

    def loss(candidate):
        logits, value, _ = candidate.forward(observation, history, mask)
        return jnp.mean(logits**2) + value**2

    gradients = eqx.filter_grad(loss)(network).conv_patch_residual
    for weight in (
        gradients.input_conv.weight,
        gradients.residual_conv_1.weight,
        gradients.residual_conv_2.weight,
        gradients.downsample_conv.weight,
        gradients.output_projection.weight,
    ):
        assert float(jnp.linalg.norm(weight)) > 0.0


@pytest.mark.parametrize("architecture", ["transformer", "conv_transformer"])
def test_competition_architectures_have_identical_external_interface(architecture):
    config = tiny_config(architecture)
    network = build_network(config, jax.random.PRNGKey(3))
    logits, value, value_logits = network.forward(
        jnp.zeros((39, 21, 21), dtype=jnp.float32),
        jnp.zeros((2, 512), dtype=jnp.float32),
        jnp.ones((3970,), dtype=jnp.bool_),
    )
    assert logits.shape == (3970,)
    assert value.shape == ()
    assert value_logits.shape == (16,)
    assert bool(jnp.all(jnp.isfinite(logits)))
    assert bool(jnp.isfinite(value))


def test_pure_and_conv_configs_are_matched_except_for_run_and_architecture():
    pure = TrainingConfig.from_toml(CONFIG_DIR / "competition_l7.toml")
    convolutional = TrainingConfig.from_toml(
        CONFIG_DIR / "competition_l7_conv.toml"
    )
    pure_values = asdict(pure)
    conv_values = asdict(convolutional)
    for field in ("run_name", "model_architecture"):
        pure_values.pop(field)
        conv_values.pop(field)
    assert pure_values == conv_values


def test_conv_checkpoint_roundtrip_and_architecture_mismatch(tmp_path):
    conv_config = tiny_config("conv_transformer")
    conv_network = build_network(conv_config, jax.random.PRNGKey(4))
    _, conv_optimizer_state = optimizer_for(conv_config, conv_network)
    key = jax.random.PRNGKey(5)
    path = tmp_path / "conv.eqx"
    _save_checkpoint(
        path,
        conv_network,
        conv_optimizer_state,
        conv_network,
        12,
        2,
        key,
    )
    skeleton = (
        conv_network,
        conv_optimizer_state,
        conv_network,
        jnp.int32(0),
        jnp.int32(0),
        key,
    )
    loaded, _, loaded_ema, iteration, stage, _ = _load_checkpoint_state(
        path, skeleton, conv_config
    )
    assert isinstance(loaded, ConvCompetitionTransformer)
    assert isinstance(loaded_ema, ConvCompetitionTransformer)
    assert int(iteration) == 12
    assert int(stage) == 2

    pure_config = tiny_config("transformer")
    pure_network = build_network(pure_config, jax.random.PRNGKey(6))
    _, pure_optimizer_state = optimizer_for(pure_config, pure_network)
    incompatible_skeleton = (
        pure_network,
        pure_optimizer_state,
        pure_network,
        jnp.int32(0),
        jnp.int32(0),
        key,
    )
    with pytest.raises(ValueError, match="model_architecture='transformer'"):
        _load_checkpoint_state(path, incompatible_skeleton, pure_config)


def test_conv_architecture_rejects_legacy_observation_schema():
    config = TrainingConfig(
        observation_schema="legacy_38", model_architecture="conv_transformer"
    )
    with pytest.raises(ValueError, match="requires observation_schema"):
        config.validate()

"""AverageJoe-style transformer with a parallel local convolutional stem."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .actions import decode_action
from .model import (
    CompetitionTransformer,
    _embed_spatial_tokens,
    _forward_transformer,
    _to_bfloat16,
)
from .observation import COMPETITION_OBSERVATION_SCHEMA


class ConvPatchResidual(eqx.Module):
    """Convert overlapping cell-level features into patch-token corrections."""

    input_conv: eqx.nn.Conv2d
    input_norm: eqx.nn.GroupNorm
    residual_norm_1: eqx.nn.GroupNorm
    residual_conv_1: eqx.nn.Conv2d
    residual_norm_2: eqx.nn.GroupNorm
    residual_conv_2: eqx.nn.Conv2d
    downsample_norm: eqx.nn.GroupNorm
    downsample_conv: eqx.nn.Conv2d
    token_norm: eqx.nn.LayerNorm
    output_projection: eqx.nn.Linear
    patch_count: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        input_channels: int,
        conv_channels: int,
        model_dim: int,
        groups: int,
        board_size: int,
        patch_size: int,
        key,
    ):
        if conv_channels % groups:
            raise ValueError("conv_channels must be divisible by groups")
        if board_size % patch_size:
            raise ValueError("board_size must be divisible by patch_size")
        if patch_size != 3:
            raise ValueError("ConvPatchResidual currently requires patch_size=3")
        keys = jax.random.split(key, 5)
        self.input_conv = eqx.nn.Conv2d(
            input_channels,
            conv_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=keys[0],
        )
        self.input_norm = eqx.nn.GroupNorm(groups, conv_channels)
        self.residual_norm_1 = eqx.nn.GroupNorm(groups, conv_channels)
        self.residual_conv_1 = eqx.nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=keys[1],
        )
        self.residual_norm_2 = eqx.nn.GroupNorm(groups, conv_channels)
        self.residual_conv_2 = eqx.nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=keys[2],
        )
        self.downsample_norm = eqx.nn.GroupNorm(groups, conv_channels)
        self.downsample_conv = eqx.nn.Conv2d(
            conv_channels,
            model_dim,
            kernel_size=3,
            stride=3,
            padding=0,
            use_bias=False,
            key=keys[3],
        )
        self.token_norm = eqx.nn.LayerNorm(model_dim)
        self.output_projection = eqx.nn.Linear(model_dim, model_dim, key=keys[4])
        self.patch_count = (board_size // patch_size) ** 2

    def __call__(self, observation):
        local = jax.nn.silu(self.input_norm(self.input_conv(observation)))

        residual = self.residual_norm_1(local)
        residual = self.residual_conv_1(jax.nn.silu(residual))
        residual = self.residual_norm_2(residual)
        residual = self.residual_conv_2(jax.nn.silu(residual))
        local = local + residual

        local = self.downsample_norm(local)
        local = self.downsample_conv(jax.nn.silu(local))
        tokens = local.reshape(local.shape[0], self.patch_count).T
        tokens = jax.nn.silu(jax.vmap(self.token_norm)(tokens))
        return jax.vmap(self.output_projection)(tokens)


class ConvCompetitionTransformer(eqx.Module):
    """Competition transformer with a convolutional patch-token residual."""

    transformer: CompetitionTransformer
    conv_patch_residual: ConvPatchResidual

    def __init__(
        self,
        *,
        board_size: int = 21,
        input_channels: int = 36,
        history_window: int = 512,
        patch_size: int = 3,
        depth: int = 7,
        model_dim: int = 448,
        heads: int = 8,
        ff_factor: int = 3,
        value_bins: int = 128,
        value_min: float = -1.0,
        value_max: float = 1.0,
        use_bf16: bool = True,
        observation_schema: str = COMPETITION_OBSERVATION_SCHEMA,
        conv_channels: int = 96,
        conv_groups: int = 12,
        key,
    ):
        if observation_schema != COMPETITION_OBSERVATION_SCHEMA:
            raise ValueError(
                "ConvCompetitionTransformer requires observation_schema="
                f"{COMPETITION_OBSERVATION_SCHEMA!r}"
            )
        self.transformer = CompetitionTransformer(
            board_size=board_size,
            input_channels=input_channels,
            history_window=history_window,
            patch_size=patch_size,
            depth=depth,
            model_dim=model_dim,
            heads=heads,
            ff_factor=ff_factor,
            value_bins=value_bins,
            value_min=value_min,
            value_max=value_max,
            use_bf16=use_bf16,
            observation_schema=observation_schema,
            key=key,
        )
        conv_key = jax.random.fold_in(key, 0xC0A5)
        self.conv_patch_residual = ConvPatchResidual(
            input_channels=input_channels,
            conv_channels=conv_channels,
            model_dim=model_dim,
            groups=conv_groups,
            board_size=board_size,
            patch_size=patch_size,
            key=conv_key,
        )

    @property
    def observation_schema(self):
        return self.transformer.observation_schema

    def spatial_token_streams(self, observation):
        """Return patch tokens and the pre-addition convolutional correction."""
        _, normalized, patch_tokens = _embed_spatial_tokens(
            self.transformer, observation
        )
        residual = (
            _to_bfloat16(self.conv_patch_residual)
            if self.transformer.use_bf16
            else self.conv_patch_residual
        )
        correction = residual(normalized)
        return patch_tokens.astype(jnp.float32), correction.astype(jnp.float32)

    def forward(self, observation, temporal_history, legal_mask):
        return _forward_transformer(
            self.transformer,
            observation,
            temporal_history,
            legal_mask,
            self.conv_patch_residual,
        )

    def __call__(
        self,
        observation,
        temporal_history,
        legal_mask,
        key,
        action_index=None,
    ):
        logits, value, value_logits = self.forward(
            observation, temporal_history, legal_mask
        )
        if action_index is None:
            action_index = jax.random.categorical(key, logits).astype(jnp.int32)
        log_probabilities = jax.nn.log_softmax(logits)
        probabilities = jax.nn.softmax(logits)
        log_probability = log_probabilities[action_index]
        entropy = -jnp.sum(probabilities * log_probabilities)
        return (
            action_index,
            decode_action(action_index),
            value,
            log_probability,
            entropy,
            value_logits,
        )


def calibrate_conv_token_rms(
    network: ConvCompetitionTransformer,
    observations,
    target_ratio: float,
) -> tuple[ConvCompetitionTransformer, dict[str, jax.Array]]:
    """Rescale the output projection once to hit a target token-RMS ratio.

    The calibration batch must contain raw augmented observations, before the
    model's schema-specific normalization. Only the final 448->448 projection
    is changed, so convolutional feature directions and the matched transformer
    backbone remain untouched.
    """
    patch_tokens, corrections = jax.vmap(network.spatial_token_streams)(
        observations
    )
    patch_rms = jnp.sqrt(jnp.mean(jnp.square(patch_tokens)))
    correction_rms_before = jnp.sqrt(jnp.mean(jnp.square(corrections)))
    multiplier = (
        jnp.asarray(target_ratio, dtype=jnp.float32)
        * patch_rms
        / jnp.maximum(correction_rms_before, jnp.finfo(jnp.float32).tiny)
    )

    projection = network.conv_patch_residual.output_projection
    scaled_projection = eqx.tree_at(
        lambda linear: (linear.weight, linear.bias),
        projection,
        (projection.weight * multiplier, projection.bias * multiplier),
    )
    calibrated = eqx.tree_at(
        lambda candidate: candidate.conv_patch_residual.output_projection,
        network,
        scaled_projection,
    )
    _, corrections_after = jax.vmap(calibrated.spatial_token_streams)(observations)
    correction_rms_after = jnp.sqrt(jnp.mean(jnp.square(corrections_after)))
    return calibrated, {
        "patch_token_rms": patch_rms,
        "correction_rms_before": correction_rms_before,
        "correction_rms_after": correction_rms_after,
        "ratio_before": correction_rms_before / patch_rms,
        "ratio_after": correction_rms_after / patch_rms,
        "projection_multiplier": multiplier,
    }

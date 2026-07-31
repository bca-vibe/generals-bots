"""Seven-layer AverageJoe-style transformer for the competition action space."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from .actions import SPATIAL_PLANES, decode_action
from .observation import LEGACY_OBSERVATION_SCHEMA, normalize_augmented_observation


def _to_bfloat16(tree):
    return jax.tree.map(
        lambda value: value.astype(jnp.bfloat16)
        if eqx.is_array(value) and jnp.issubdtype(value.dtype, jnp.floating)
        else value,
        tree,
    )


class MultiHeadSelfAttention(eqx.Module):
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    output: eqx.nn.Linear
    heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, model_dim: int, heads: int, *, key):
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = model_dim // heads
        keys = jax.random.split(key, 4)
        self.query = eqx.nn.Linear(model_dim, model_dim, key=keys[0])
        self.key = eqx.nn.Linear(model_dim, model_dim, key=keys[1])
        self.value = eqx.nn.Linear(model_dim, model_dim, key=keys[2])
        self.output = eqx.nn.Linear(model_dim, model_dim, key=keys[3])

    def __call__(self, tokens):
        token_count = tokens.shape[0]
        query = jax.vmap(self.query)(tokens).reshape(token_count, self.heads, self.head_dim)
        key = jax.vmap(self.key)(tokens).reshape(token_count, self.heads, self.head_dim)
        value = jax.vmap(self.value)(tokens).reshape(token_count, self.heads, self.head_dim)
        query, key, value = (
            jnp.transpose(item, (1, 0, 2)) for item in (query, key, value)
        )
        scores = query @ jnp.transpose(key, (0, 2, 1)) / math.sqrt(self.head_dim)
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(query.dtype)
        attended = weights @ value
        attended = jnp.transpose(attended, (1, 0, 2)).reshape(token_count, -1)
        return jax.vmap(self.output)(attended)


class TransformerBlock(eqx.Module):
    attention_norm: eqx.nn.LayerNorm
    attention: MultiHeadSelfAttention
    feedforward_norm: eqx.nn.LayerNorm
    feedforward_in: eqx.nn.Linear
    feedforward_out: eqx.nn.Linear

    def __init__(self, model_dim: int, heads: int, ff_factor: int, *, key):
        attention_key, ff_in_key, ff_out_key = jax.random.split(key, 3)
        self.attention_norm = eqx.nn.LayerNorm(model_dim)
        self.attention = MultiHeadSelfAttention(model_dim, heads, key=attention_key)
        self.feedforward_norm = eqx.nn.LayerNorm(model_dim)
        self.feedforward_in = eqx.nn.Linear(model_dim, ff_factor * model_dim, key=ff_in_key)
        self.feedforward_out = eqx.nn.Linear(ff_factor * model_dim, model_dim, key=ff_out_key)

    def __call__(self, tokens):
        tokens = tokens + self.attention(jax.vmap(self.attention_norm)(tokens))
        hidden = jax.vmap(self.feedforward_norm)(tokens)
        hidden = jax.nn.silu(jax.vmap(self.feedforward_in)(hidden))
        return tokens + jax.vmap(self.feedforward_out)(hidden)


class TemporalEncoder(eqx.Module):
    army_in: eqx.nn.Linear
    army_out: eqx.nn.Linear
    land_in: eqx.nn.Linear
    land_out: eqx.nn.Linear

    def __init__(self, temporal_window: int, model_dim: int, *, key):
        keys = jax.random.split(key, 4)
        self.army_in = eqx.nn.Linear(temporal_window, 512, key=keys[0])
        self.army_out = eqx.nn.Linear(512, model_dim, key=keys[1])
        self.land_in = eqx.nn.Linear(temporal_window, 512, key=keys[2])
        self.land_out = eqx.nn.Linear(512, model_dim, key=keys[3])

    def __call__(self, histories):
        army = self.army_out(jax.nn.silu(self.army_in(histories[0] / 50.0)))
        land = self.land_out(jax.nn.silu(self.land_in(histories[1] / 50.0)))
        return jnp.stack([army, land])


class CompetitionTransformer(eqx.Module):
    patch_embedding: eqx.nn.Linear
    value_token: jax.Array
    position_embedding: jax.Array
    temporal_type_embedding: jax.Array
    temporal_encoder: TemporalEncoder
    blocks: tuple[TransformerBlock, ...]
    output_norm: eqx.nn.LayerNorm
    spatial_policy_head: eqx.nn.Linear
    pass_head: eqx.nn.Linear
    value_head: eqx.nn.Linear
    board_size: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    model_dim: int = eqx.field(static=True)
    value_bins: int = eqx.field(static=True)
    value_min: float = eqx.field(static=True)
    value_max: float = eqx.field(static=True)
    use_bf16: bool = eqx.field(static=True)
    observation_schema: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        board_size: int = 21,
        input_channels: int = 38,
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
        observation_schema: str = LEGACY_OBSERVATION_SCHEMA,
        key,
    ):
        if board_size % patch_size:
            raise ValueError("board_size must be divisible by patch_size")
        self.board_size = board_size
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.value_bins = value_bins
        self.value_min = value_min
        self.value_max = value_max
        self.use_bf16 = use_bf16
        self.observation_schema = observation_schema
        keys = jax.random.split(key, depth + 9)
        patch_dim = input_channels * patch_size * patch_size
        self.patch_embedding = eqx.nn.Linear(patch_dim, model_dim, key=keys[0])
        patch_count = (board_size // patch_size) ** 2
        token_count = patch_count + 3
        self.value_token = jax.random.normal(keys[1], (1, model_dim)) * 0.02
        self.position_embedding = jax.random.truncated_normal(
            keys[2], -2.0, 2.0, (token_count, model_dim)
        ) * 0.1
        self.temporal_type_embedding = jax.random.normal(keys[3], (2, model_dim)) * 0.02
        self.temporal_encoder = TemporalEncoder(history_window, model_dim, key=keys[4])
        self.blocks = tuple(
            TransformerBlock(model_dim, heads, ff_factor, key=keys[5 + index])
            for index in range(depth)
        )
        self.output_norm = eqx.nn.LayerNorm(model_dim)
        head_key = 5 + depth
        self.spatial_policy_head = eqx.nn.Linear(
            model_dim, SPATIAL_PLANES * patch_size * patch_size, key=keys[head_key]
        )
        self.pass_head = eqx.nn.Linear(model_dim, 1, key=keys[head_key + 1])
        self.value_head = eqx.nn.Linear(model_dim, value_bins, key=keys[head_key + 2])

    def forward(self, observation, temporal_history, legal_mask):
        return _forward_transformer(self, observation, temporal_history, legal_mask)

    def __call__(self, observation, temporal_history, legal_mask, key, action_index=None):
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


def greedy_action(model, observation, temporal_history, legal_mask):
    logits, _, _ = model.forward(observation, temporal_history, legal_mask)
    return decode_action(jnp.argmax(logits))


def _embed_spatial_tokens(transformer, observation):
    """Return the compute-cast model, normalized input, and patch tokens."""
    net = _to_bfloat16(transformer) if transformer.use_bf16 else transformer
    observation = normalize_augmented_observation(
        observation, transformer.observation_schema
    )
    if transformer.use_bf16:
        observation = observation.astype(jnp.bfloat16)

    patch_grid = transformer.board_size // transformer.patch_size
    patches = observation.reshape(
        observation.shape[0],
        patch_grid,
        transformer.patch_size,
        patch_grid,
        transformer.patch_size,
    )
    patches = patches.transpose(1, 3, 0, 2, 4).reshape(patch_grid**2, -1)
    return net, observation, jax.vmap(net.patch_embedding)(patches)


def _forward_transformer(
    transformer,
    observation,
    temporal_history,
    legal_mask,
    patch_residual=None,
):
    """Run the shared transformer, optionally adding a spatial-token correction."""
    net, observation, patch_tokens = _embed_spatial_tokens(
        transformer, observation
    )
    if transformer.use_bf16:
        temporal_history = temporal_history.astype(jnp.bfloat16)

    patch_grid = transformer.board_size // transformer.patch_size
    if patch_residual is not None:
        residual = (
            _to_bfloat16(patch_residual)
            if transformer.use_bf16
            else patch_residual
        )
        patch_tokens = patch_tokens + residual(observation)

    history_tokens = (
        net.temporal_encoder(temporal_history) + net.temporal_type_embedding
    )
    tokens = jnp.concatenate([net.value_token, history_tokens, patch_tokens])
    tokens = tokens + net.position_embedding
    for block in net.blocks:
        tokens = block(tokens)
    tokens = jax.vmap(net.output_norm)(tokens)

    value_logits = net.value_head(tokens[0]).astype(jnp.float32)
    bin_centers = jnp.linspace(
        transformer.value_min, transformer.value_max, transformer.value_bins
    )
    value = jnp.sum(jax.nn.softmax(value_logits) * bin_centers)

    patch_logits = jax.vmap(net.spatial_policy_head)(tokens[3:]).astype(jnp.float32)
    spatial_logits = patch_logits.reshape(
        patch_grid,
        patch_grid,
        SPATIAL_PLANES,
        transformer.patch_size,
        transformer.patch_size,
    )
    spatial_logits = spatial_logits.transpose(2, 0, 3, 1, 4).reshape(-1)
    pass_logit = net.pass_head(tokens[0]).astype(jnp.float32)
    logits = jnp.concatenate([spatial_logits, pass_logit])
    logits = jnp.where(legal_mask, logits, -1e9)
    return logits, value, value_logits

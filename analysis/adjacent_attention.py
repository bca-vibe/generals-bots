"""Analyze adjacent spatial-token attention with TransformerLens."""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from generals.core.game import get_observation
from generals.training.actions import decode_action, legal_action_mask
from generals.training.config import TrainingConfig
from generals.training.conv_model import ConvCompetitionTransformer
from generals.training.model import CompetitionTransformer
from generals.training.observation import (
    augment_observation,
    init_observation_memory,
    normalize_augmented_observation,
    reset_finished_memory,
    temporal_input,
)
from generals.training.train import (
    _learning_rate,
    _load_checkpoint_state,
    build_network,
    make_environment,
)


def load_checkpoint(config: TrainingConfig, path: Path | None):
    """Load EMA parameters, or return the identically initialized network."""
    key = jax.random.PRNGKey(config.seed)
    key, network_key = jax.random.split(key)
    network = build_network(config, network_key)
    if path is None:
        return network
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(partial(_learning_rate, config)),
    )
    optimizer_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    skeleton = (network, optimizer_state, network, jnp.int32(0), jnp.int32(0), key)
    _, _, ema_network, _, _, _ = _load_checkpoint_state(
        path, skeleton, config
    )
    return ema_network


def concatenate_players(player_zero, player_one):
    return jax.tree.map(
        lambda left, right: jnp.concatenate([left, right]), player_zero, player_one
    )


def batched_memory(config: TrainingConfig, count: int):
    memory = init_observation_memory(
        config.pad_to, config.history_size, config.temporal_window
    )
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (count, *value.shape)), memory
    )


def transformer_backbone(network):
    """Return the shared transformer from either supported architecture."""
    if isinstance(network, ConvCompetitionTransformer):
        return network.transformer
    return network


def preblock_tokens(network, observation, histories):
    """Reproduce model.forward through position addition, in float32."""
    backbone = transformer_backbone(network)
    observation = normalize_augmented_observation(
        observation, backbone.observation_schema
    ).astype(jnp.float32)
    histories = histories.astype(jnp.float32)
    patch_grid = backbone.board_size // backbone.patch_size
    patches = observation.reshape(
        observation.shape[0],
        patch_grid,
        backbone.patch_size,
        patch_grid,
        backbone.patch_size,
    )
    patches = patches.transpose(1, 3, 0, 2, 4).reshape(patch_grid**2, -1)
    patch_tokens = jax.vmap(backbone.patch_embedding)(patches)
    if isinstance(network, ConvCompetitionTransformer):
        patch_tokens = patch_tokens + network.conv_patch_residual(observation)
    history_tokens = (
        backbone.temporal_encoder(histories) + backbone.temporal_type_embedding
    )
    return (
        jnp.concatenate([backbone.value_token, history_tokens, patch_tokens])
        + backbone.position_embedding
    )


def collect_held_out_inputs(config, policy, n_maps, sample_turns, seed):
    """Collect observations from stochastic EMA self-play on fresh final-stage maps."""
    environment = make_environment(config, config.curriculum[-1], pool_size=max(64, n_maps))
    key = jax.random.PRNGKey(seed)
    key, pool_key = jax.random.split(key)
    pool, _ = environment.reset(pool_key)
    states = jax.tree.map(lambda value: value[:n_maps], pool)
    memory = batched_memory(config, 2 * n_maps)

    @eqx.filter_jit
    def step_batch(states, memory, rng):
        obs_zero = jax.vmap(lambda state: get_observation(state, 0))(states)
        obs_one = jax.vmap(lambda state: get_observation(state, 1))(states)
        observations = concatenate_players(obs_zero, obs_one)
        board_masks = jnp.concatenate([states.board_mask, states.board_mask])
        augmented, next_memory = jax.vmap(
            lambda observation, current_memory, board_mask: augment_observation(
                observation,
                current_memory,
                board_mask,
                config.observation_schema,
            )
        )(
            observations, memory, board_masks
        )
        histories = temporal_input(next_memory)
        masks = jax.vmap(legal_action_mask)(observations, board_masks)
        split_keys = jax.random.split(rng, 2 * n_maps + 1)
        logits = jax.vmap(
            lambda obs, history, mask: policy.forward(obs, history, mask)[0]
        )(augmented, histories, masks)
        indices = jax.vmap(jax.random.categorical)(split_keys[1:], logits)
        actions = jax.vmap(decode_action)(indices)
        environment_actions = jnp.stack([actions[:n_maps], actions[n_maps:]], axis=1)
        timesteps, next_states = jax.vmap(
            lambda state, action: environment.step(state, action, pool)
        )(states, environment_actions)
        finished = timesteps.terminated | timesteps.truncated
        next_memory = reset_finished_memory(
            next_memory, jnp.concatenate([finished, finished])
        )
        return next_states, next_memory, split_keys[0], augmented, histories

    observations_out, histories_out, groups_out, turns_out = [], [], [], []
    selected = set(sample_turns)
    for turn in range(max(sample_turns) + 1):
        states, memory, key, augmented, histories = step_batch(states, memory, key)
        if turn in selected:
            observations_out.append(np.asarray(jax.device_get(augmented)))
            histories_out.append(np.asarray(jax.device_get(histories)))
            groups_out.append(np.tile(np.arange(n_maps), 2))
            turns_out.append(np.full(2 * n_maps, turn, dtype=np.int32))
    return (
        np.concatenate(observations_out),
        np.concatenate(histories_out),
        np.concatenate(groups_out),
        np.concatenate(turns_out),
    )


def make_hooked_transformer(
    network: CompetitionTransformer | ConvCompetitionTransformer,
):
    """Copy Equinox transformer block weights into a TransformerLens model."""
    network = transformer_backbone(network)
    config = HookedTransformerConfig(
        n_layers=len(network.blocks),
        d_model=network.model_dim,
        d_head=network.blocks[0].attention.head_dim,
        n_heads=network.blocks[0].attention.heads,
        d_mlp=network.blocks[0].feedforward_in.out_features,
        n_ctx=(network.board_size // network.patch_size) ** 2 + 3,
        d_vocab=1,
        d_vocab_out=1,
        act_fn="silu",
        normalization_type="LN",
        attention_dir="bidirectional",
        default_prepend_bos=False,
        device="cpu",
        dtype=torch.float32,
    )
    hooked = HookedTransformer(config, move_to_device=False)

    def tensor(value):
        array = np.asarray(jax.device_get(value), dtype=np.float32)
        return torch.from_numpy(array)

    with torch.no_grad():
        for source, target in zip(network.blocks, hooked.blocks):
            target.ln1.w.copy_(tensor(source.attention_norm.weight))
            target.ln1.b.copy_(tensor(source.attention_norm.bias))
            target.ln2.w.copy_(tensor(source.feedforward_norm.weight))
            target.ln2.b.copy_(tensor(source.feedforward_norm.bias))
            heads = source.attention.heads
            head_dim = source.attention.head_dim
            for source_linear, weight, bias in (
                (source.attention.query, target.attn.W_Q, target.attn.b_Q),
                (source.attention.key, target.attn.W_K, target.attn.b_K),
                (source.attention.value, target.attn.W_V, target.attn.b_V),
            ):
                weight.copy_(
                    tensor(source_linear.weight)
                    .reshape(heads, head_dim, network.model_dim)
                    .permute(0, 2, 1)
                )
                bias.copy_(tensor(source_linear.bias).reshape(heads, head_dim))
            target.attn.W_O.copy_(
                tensor(source.attention.output.weight)
                .T.reshape(heads, head_dim, network.model_dim)
            )
            target.attn.b_O.copy_(tensor(source.attention.output.bias))
            target.mlp.W_in.copy_(tensor(source.feedforward_in.weight).T)
            target.mlp.b_in.copy_(tensor(source.feedforward_in.bias))
            target.mlp.W_out.copy_(tensor(source.feedforward_out.weight).T)
            target.mlp.b_out.copy_(tensor(source.feedforward_out.bias))
        hooked.ln_final.w.copy_(tensor(network.output_norm.weight))
        hooked.ln_final.b.copy_(tensor(network.output_norm.bias))
    hooked.eval()
    return hooked


def direct_jax_patterns(network, tokens, layers):
    """Independent pattern calculation used to validate TransformerLens conversion."""
    network = transformer_backbone(network)
    patterns = []
    for block in network.blocks[:layers]:
        normalized = jax.vmap(block.attention_norm)(tokens)
        count = normalized.shape[0]
        query = jax.vmap(block.attention.query)(normalized).reshape(
            count, block.attention.heads, block.attention.head_dim
        )
        key = jax.vmap(block.attention.key)(normalized).reshape(
            count, block.attention.heads, block.attention.head_dim
        )
        query, key = jnp.transpose(query, (1, 0, 2)), jnp.transpose(key, (1, 0, 2))
        scores = query @ jnp.transpose(key, (0, 2, 1)) / np.sqrt(block.attention.head_dim)
        patterns.append(jax.nn.softmax(scores.astype(jnp.float32), axis=-1))
        tokens = block(tokens)
    return jnp.stack(patterns)


def capture_patterns(network, augmented, histories, layers, batch_size=128):
    tokenize = eqx.filter_jit(
        jax.vmap(lambda obs, hist: preblock_tokens(network, obs, hist))
    )
    hooked = make_hooked_transformer(network)
    captured = [[] for _ in range(layers)]
    equivalence_error = None
    for start in range(0, len(augmented), batch_size):
        stop = min(start + batch_size, len(augmented))
        tokens = np.asarray(
            jax.device_get(tokenize(augmented[start:stop], histories[start:stop])),
            dtype=np.float32,
        )
        _, cache = hooked.run_with_cache(
            torch.from_numpy(tokens),
            start_at_layer=0,
            stop_at_layer=layers,
            return_type=None,
            names_filter=lambda name: name.endswith("attn.hook_pattern"),
        )
        for layer in range(layers):
            captured[layer].append(
                cache[f"blocks.{layer}.attn.hook_pattern"].detach().cpu().numpy()
            )
        if equivalence_error is None:
            expected = np.asarray(direct_jax_patterns(network, jnp.asarray(tokens[0]), layers))
            actual = np.stack([captured[layer][0][0] for layer in range(layers)])
            equivalence_error = float(np.max(np.abs(expected - actual)))
    return np.stack([np.concatenate(items) for items in captured]), equivalence_error


def spatial_masks(grid=7):
    positions = np.stack(np.unravel_index(np.arange(grid * grid), (grid, grid)), axis=1)
    delta = positions[:, None, :] - positions[None, :, :]
    manhattan = np.abs(delta).sum(axis=-1)
    chebyshev = np.abs(delta).max(axis=-1)
    return (
        manhattan,
        manhattan == 1,
        (chebyshev == 1) & (manhattan > 0),
        manhattan > 0,
    )


def bootstrap_ratio(numerator, denominator, groups, seed, draws=4000):
    unique = np.unique(groups)
    group_num = np.array([numerator[groups == group].mean() for group in unique])
    group_den = np.array([denominator[groups == group].mean() for group in unique])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(draws, len(unique)))
    ratios = group_num[sampled].mean(axis=1) / group_den[sampled].mean(axis=1)
    return np.quantile(ratios, [0.025, 0.975])


def summarize_patterns(patterns, groups, label, seed):
    """Return one row per head; patterns are [layer,sample,head,query,key]."""
    manhattan, adjacent4, adjacent8, nonself = spatial_masks()
    spatial = patterns[..., 3:, 3:]
    special_mass = patterns[..., 3:, :3].sum(axis=-1).mean(axis=(1, 3))
    self_mass = np.diagonal(spatial, axis1=-2, axis2=-1).mean(axis=(1, 3))
    nonself_mass = (spatial * nonself).sum(axis=-1)
    expected4 = nonself_mass * adjacent4.sum(axis=-1)[None, None, None, :] / 48.0
    expected8 = nonself_mass * adjacent8.sum(axis=-1)[None, None, None, :] / 48.0
    mass4 = (spatial * adjacent4).sum(axis=-1)
    mass8 = (spatial * adjacent8).sum(axis=-1)
    no_self = np.where(nonself[None, None, None, :, :], spatial, -np.inf)
    top_keys = no_self.argmax(axis=-1)
    query_indices = np.arange(49)[None, None, None, :]
    top4 = adjacent4[query_indices, top_keys].mean(axis=(1, 3))
    top8 = adjacent8[query_indices, top_keys].mean(axis=(1, 3))
    distance_num = (spatial * manhattan).sum(axis=(-1, -2))
    distance_den = (spatial * nonself).sum(axis=(-1, -2))
    mean_distance = (distance_num / distance_den).mean(axis=1)

    rows = []
    for layer in range(patterns.shape[0]):
        for head in range(patterns.shape[2]):
            numerator4 = mass4[layer, :, head].mean(axis=-1)
            denominator4 = expected4[layer, :, head].mean(axis=-1)
            numerator8 = mass8[layer, :, head].mean(axis=-1)
            denominator8 = expected8[layer, :, head].mean(axis=-1)
            ci4 = bootstrap_ratio(numerator4, denominator4, groups, seed + layer * 100 + head)
            ci8 = bootstrap_ratio(
                numerator8, denominator8, groups, seed + 1000 + layer * 100 + head
            )
            rows.append(
                {
                    "checkpoint": label,
                    "layer": layer,
                    "head": head,
                    "adjacent4_mass": float(numerator4.mean()),
                    "adjacent4_expected": float(denominator4.mean()),
                    "adjacent4_enrichment": float(numerator4.mean() / denominator4.mean()),
                    "adjacent4_ci_low": float(ci4[0]),
                    "adjacent4_ci_high": float(ci4[1]),
                    "adjacent8_mass": float(numerator8.mean()),
                    "adjacent8_expected": float(denominator8.mean()),
                    "adjacent8_enrichment": float(numerator8.mean() / denominator8.mean()),
                    "adjacent8_ci_low": float(ci8[0]),
                    "adjacent8_ci_high": float(ci8[1]),
                    "top_spatial_key_adjacent4": float(top4[layer, head]),
                    "top_spatial_key_adjacent8": float(top8[layer, head]),
                    "mean_attended_manhattan_distance": float(mean_distance[layer, head]),
                    "self_mass": float(self_mass[layer, head]),
                    "special_token_mass": float(special_mass[layer, head]),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="generals/training/configs/smoke_8xh100.toml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/smoke_8xh100")
    parser.add_argument("--output-dir", default="runs/smoke_8xh100/attention_analysis")
    parser.add_argument("--maps", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    config = TrainingConfig.from_toml(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)
    specs = [
        ("initial", None),
        ("iteration_540", checkpoint_dir / "checkpoint_000540.eqx"),
        ("iteration_880", checkpoint_dir / "checkpoint_000880.eqx"),
        ("iteration_1260", checkpoint_dir / "checkpoint_001260.eqx"),
    ]
    networks = {label: load_checkpoint(config, path) for label, path in specs}
    sample_turns = (0, 5, 10, 20, 35, 49, 50, 75, 100, 150, 225, 300, 400)
    augmented, histories, groups, _ = collect_held_out_inputs(
        config, networks["iteration_1260"], args.maps, sample_turns, args.seed
    )

    rows, errors = [], {}
    for label, _ in specs:
        patterns, errors[label] = capture_patterns(
            networks[label], augmented, histories, args.layers
        )
        rows.extend(summarize_patterns(patterns, groups, label, args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "head_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    latest = sorted(
        (row for row in rows if row["checkpoint"] == "iteration_1260"),
        key=lambda row: row["adjacent4_enrichment"],
        reverse=True,
    )
    initial = {
        (row["layer"], row["head"]): row
        for row in rows
        if row["checkpoint"] == "initial"
    }
    summary = {
        "checkpoint": str(checkpoint_dir / "checkpoint_001260.eqx"),
        "uses_ema_parameters": True,
        "held_out_map_seed": args.seed,
        "held_out_maps": args.maps,
        "sample_turns": sample_turns,
        "observations": int(len(augmented)),
        "transformer_lens_equivalence_max_abs_error": errors,
        "null_definition": (
            "Each query's non-self spatial attention mass redistributed uniformly "
            "over the other 48 spatial keys, preserving border degree."
        ),
        "top_latest_heads": [
            {
                **row,
                "initial_adjacent4_enrichment": initial[(row["layer"], row["head"])][
                    "adjacent4_enrichment"
                ],
            }
            for row in latest[:10]
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

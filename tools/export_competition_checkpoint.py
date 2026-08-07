"""Export a raw or EMA policy from a full checkpoint as a competition bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from generals.training.config import TrainingConfig
from generals.training.train import _learning_rate, build_network


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _add_linear(weights: dict, prefix: str, linear) -> None:
    weights[prefix + ".weight"] = linear.weight
    weights[prefix + ".bias"] = linear.bias


def _add_norm(weights: dict, prefix: str, norm) -> None:
    weights[prefix + ".weight"] = norm.weight
    weights[prefix + ".bias"] = norm.bias


def _extract_weights(ema) -> dict:
    transformer = ema.transformer
    conv = ema.conv_patch_residual
    weights = {
        "value_token": transformer.value_token,
        "position_embedding": transformer.position_embedding,
        "temporal_type_embedding": transformer.temporal_type_embedding,
        "conv.input_conv.weight": conv.input_conv.weight,
        "conv.residual_conv_1.weight": conv.residual_conv_1.weight,
        "conv.residual_conv_2.weight": conv.residual_conv_2.weight,
        "conv.downsample_conv.weight": conv.downsample_conv.weight,
    }
    _add_linear(weights, "patch_embedding", transformer.patch_embedding)
    for branch in ("army", "land"):
        _add_linear(
            weights,
            f"temporal_encoder.{branch}_in",
            getattr(transformer.temporal_encoder, f"{branch}_in"),
        )
        _add_linear(
            weights,
            f"temporal_encoder.{branch}_out",
            getattr(transformer.temporal_encoder, f"{branch}_out"),
        )
    for index, block in enumerate(transformer.blocks):
        prefix = f"blocks.{index}"
        _add_norm(weights, prefix + ".attention_norm", block.attention_norm)
        for name in ("query", "key", "value", "output"):
            _add_linear(
                weights,
                prefix + f".attention.{name}",
                getattr(block.attention, name),
            )
        _add_norm(weights, prefix + ".feedforward_norm", block.feedforward_norm)
        _add_linear(weights, prefix + ".feedforward_in", block.feedforward_in)
        _add_linear(weights, prefix + ".feedforward_out", block.feedforward_out)
    _add_norm(weights, "output_norm", transformer.output_norm)
    _add_linear(weights, "spatial_policy_head", transformer.spatial_policy_head)
    _add_linear(weights, "pass_head", transformer.pass_head)
    if ema.build_kind_head is not None:
        _add_linear(weights, "build_kind_head", ema.build_kind_head)
    for name in (
        "input_norm",
        "residual_norm_1",
        "residual_norm_2",
        "downsample_norm",
        "token_norm",
    ):
        _add_norm(weights, "conv." + name, getattr(conv, name))
    _add_linear(weights, "conv.output_projection", conv.output_projection)
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--source", required=True, help="Provenance URI for metadata")
    parser.add_argument(
        "--policy",
        choices=("raw", "ema"),
        default="ema",
        help="Which checkpoint parameter tree to package",
    )
    parser.add_argument(
        "--template-dir", default="competition/agents/conv_1313"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != args.expected_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: {checkpoint_sha256} != "
            f"{args.expected_sha256}"
        )

    config = TrainingConfig.from_toml(args.config)
    key = jax.random.PRNGKey(config.seed)
    key, network_key = jax.random.split(key)
    network = build_network(config, network_key)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(partial(_learning_rate, config)),
    )
    optimizer_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    skeleton = (
        network,
        optimizer_state,
        network,
        jnp.int32(0),
        jnp.int32(0),
        key,
    )
    raw, _, ema, iteration, stage, _ = eqx.tree_deserialise_leaves(
        checkpoint, skeleton
    )
    selected_policy = raw if args.policy == "raw" else ema

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    template_dir = Path(args.template_dir)
    for name in ("main.py", "bot.py", "run.sh", "build.sh"):
        shutil.copy2(template_dir / name, output_dir / name)

    output = output_dir / "weights.npz"
    packed = {
        name: np.asarray(value.astype(jnp.bfloat16)).view(np.uint16)
        for name, value in _extract_weights(selected_policy).items()
    }
    np.savez_compressed(output, **packed)
    metadata = {
        "architecture": config.model_architecture,
        "checkpoint": args.source,
        "checkpoint_sha256": checkpoint_sha256,
        "curriculum_stage": int(stage),
        "iteration": int(iteration),
        "observation_schema": config.observation_schema,
        "policy": args.policy,
        "weights_bytes": output.stat().st_size,
        "weights_sha256": _sha256(output),
    }
    (output_dir / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()

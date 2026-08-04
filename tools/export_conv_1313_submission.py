"""Export the final iteration-1313 convolutional EMA policy for competition."""

from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from generals.training.config import TrainingConfig
from generals.training.train import _learning_rate, build_network

ROOT = Path(__file__).resolve().parents[1]
RUN = "arch_ab_d448_8xh100_5h_retry1_20260803"
CONFIG = ROOT / "generals/training/configs/arch_ab_d448_8xh100_5h_conv.toml"
CHECKPOINT = ROOT / "checkpoints/huggingface/runs" / RUN / "branches/conv/terminal.eqx"
OUTPUT = ROOT / "competition/agents/conv_1313/weights.npz"
METADATA = ROOT / "competition/agents/conv_1313/export_metadata.json"
EXPECTED_CHECKPOINT_SHA256 = "23cf2c76ea348ee67b3c0dd796920a03cc2a1268e641d323dd926a5803306bd2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_linear(weights: dict, prefix: str, linear) -> None:
    weights[prefix + ".weight"] = linear.weight
    weights[prefix + ".bias"] = linear.bias


def _add_norm(weights: dict, prefix: str, norm) -> None:
    weights[prefix + ".weight"] = norm.weight
    weights[prefix + ".bias"] = norm.bias


def main() -> None:
    checkpoint_sha256 = _sha256(CHECKPOINT)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint_sha256} != {EXPECTED_CHECKPOINT_SHA256}")

    config = TrainingConfig.from_toml(CONFIG)
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
    _, _, ema, iteration, stage, _ = eqx.tree_deserialise_leaves(CHECKPOINT, skeleton)
    if int(iteration) != 1313 or int(stage) != 4:
        raise ValueError(f"Unexpected checkpoint state: iteration={iteration}, stage={stage}")

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
    _add_linear(weights, "temporal_encoder.army_in", transformer.temporal_encoder.army_in)
    _add_linear(weights, "temporal_encoder.army_out", transformer.temporal_encoder.army_out)
    _add_linear(weights, "temporal_encoder.land_in", transformer.temporal_encoder.land_in)
    _add_linear(weights, "temporal_encoder.land_out", transformer.temporal_encoder.land_out)
    for index, block in enumerate(transformer.blocks):
        prefix = f"blocks.{index}"
        _add_norm(weights, prefix + ".attention_norm", block.attention_norm)
        _add_linear(weights, prefix + ".attention.query", block.attention.query)
        _add_linear(weights, prefix + ".attention.key", block.attention.key)
        _add_linear(weights, prefix + ".attention.value", block.attention.value)
        _add_linear(weights, prefix + ".attention.output", block.attention.output)
        _add_norm(weights, prefix + ".feedforward_norm", block.feedforward_norm)
        _add_linear(weights, prefix + ".feedforward_in", block.feedforward_in)
        _add_linear(weights, prefix + ".feedforward_out", block.feedforward_out)
    _add_norm(weights, "output_norm", transformer.output_norm)
    _add_linear(weights, "spatial_policy_head", transformer.spatial_policy_head)
    _add_linear(weights, "pass_head", transformer.pass_head)

    for name in (
        "input_norm",
        "residual_norm_1",
        "residual_norm_2",
        "downsample_norm",
        "token_norm",
    ):
        _add_norm(weights, "conv." + name, getattr(conv, name))
    _add_linear(weights, "conv.output_projection", conv.output_projection)

    # NumPy can load these arrays without ml_dtypes in the competition image.
    packed = {name: np.asarray(value.astype(jnp.bfloat16)).view(np.uint16) for name, value in weights.items()}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **packed)
    metadata = {
        "architecture": config.model_architecture,
        "checkpoint": (f"bca-vibe/generals-bot/runs/{RUN}/branches/conv/terminal.eqx"),
        "checkpoint_sha256": checkpoint_sha256,
        "curriculum_stage": int(stage),
        "iteration": int(iteration),
        "observation_schema": config.observation_schema,
        "policy": "ema",
        "weights_bytes": OUTPUT.stat().st_size,
        "weights_sha256": _sha256(OUTPUT),
    }
    METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024**2:.2f} MiB, sha256={metadata['weights_sha256']})")


if __name__ == "__main__":
    main()

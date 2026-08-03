"""Export the iteration-1260 EMA policy to the competition submission."""

from __future__ import annotations

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
CONFIG = ROOT / "generals/training/configs/smoke_8xh100.toml"
CHECKPOINT = ROOT / "checkpoints/smoke_8xh100/checkpoint_001260.eqx"
OUTPUT = ROOT / "competition/agents/smoke_1260_baseline/weights.npz"


def _add_linear(weights: dict, prefix: str, linear) -> None:
    weights[prefix + ".weight"] = linear.weight
    weights[prefix + ".bias"] = linear.bias


def _add_norm(weights: dict, prefix: str, norm) -> None:
    weights[prefix + ".weight"] = norm.weight
    weights[prefix + ".bias"] = norm.bias


def main() -> None:
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
    if int(iteration) != 1260 or int(stage) != 4:
        raise ValueError(f"Unexpected checkpoint state: iteration={iteration}, stage={stage}")

    weights = {
        "value_token": ema.value_token,
        "position_embedding": ema.position_embedding,
        "temporal_type_embedding": ema.temporal_type_embedding,
    }
    _add_linear(weights, "patch_embedding", ema.patch_embedding)
    _add_linear(weights, "temporal_encoder.army_in", ema.temporal_encoder.army_in)
    _add_linear(weights, "temporal_encoder.army_out", ema.temporal_encoder.army_out)
    _add_linear(weights, "temporal_encoder.land_in", ema.temporal_encoder.land_in)
    _add_linear(weights, "temporal_encoder.land_out", ema.temporal_encoder.land_out)
    for index, block in enumerate(ema.blocks):
        prefix = f"blocks.{index}"
        _add_norm(weights, prefix + ".attention_norm", block.attention_norm)
        _add_linear(weights, prefix + ".attention.query", block.attention.query)
        _add_linear(weights, prefix + ".attention.key", block.attention.key)
        _add_linear(weights, prefix + ".attention.value", block.attention.value)
        _add_linear(weights, prefix + ".attention.output", block.attention.output)
        _add_norm(weights, prefix + ".feedforward_norm", block.feedforward_norm)
        _add_linear(weights, prefix + ".feedforward_in", block.feedforward_in)
        _add_linear(weights, prefix + ".feedforward_out", block.feedforward_out)
    _add_norm(weights, "output_norm", ema.output_norm)
    _add_linear(weights, "spatial_policy_head", ema.spatial_policy_head)
    _add_linear(weights, "pass_head", ema.pass_head)

    # JAX's bfloat16 arrays are stored as their raw uint16 words so NumPy can
    # load them without depending on ml_dtypes at runtime.
    packed = {
        name: np.asarray(value.astype(jnp.bfloat16)).view(np.uint16)
        for name, value in weights.items()
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **packed)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()

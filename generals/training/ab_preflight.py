"""Concurrent eight-GPU initialization preflight for the architecture A/B run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path


def run_branch(config_path: Path, output: Path, *, full_training: bool = False) -> None:
    # Import JAX only after the child has inherited its branch-specific
    # CUDA_VISIBLE_DEVICES. Importing the training stack in the eight-GPU
    # parent initializes JAX and reserves most of physical GPU 0 before either
    # four-GPU worker starts.
    import jax
    import jax.numpy as jnp

    from .config import TrainingConfig
    from .conv_model import ConvCompetitionTransformer, calibrate_conv_token_rms
    from .train import _tree_sha256, build_network, train

    config = TrainingConfig.from_toml(config_path)
    if jax.device_count() != 4:
        raise RuntimeError(f"Expected four visible H100s, found {jax.devices()}")
    if full_training:
        run_name = f"preflight_{config.model_architecture}"
        config = replace(
            config,
            run_name=run_name,
            output_dir=str(output.parent / "runs"),
            wandb_project=None,
            wandb_group=None,
            wandb_run_id=None,
            wandb_tags=(),
            num_iterations=1,
            eval_every=1,
            eval_games=16,
            latest_checkpoint_every=1,
            checkpoint_every=1,
            league_eval_after_training=False,
            league_eval_every=1,
            league_eval_maps=8,
            league_eval_policies=("raw", "ema"),
            league_opponents=("random",),
            reset_pool_every=0,
        )
        train(config)
        run_dir = config.run_dir
        initialization = json.loads(
            (run_dir / "initialization.json").read_text(encoding="utf-8")
        )
        checkpoint = json.loads(
            (run_dir / "latest_checkpoint.json").read_text(encoding="utf-8")
        )
        league = json.loads(
            (run_dir / "league_000001.json").read_text(encoding="utf-8")
        )
        record = {
            "architecture": config.model_architecture,
            "devices": [str(device) for device in jax.devices()],
            "transformer_trunk_sha256": initialization["transformer_trunk_sha256"],
            "checkpoint_iteration": checkpoint["iteration"],
            "checkpoint_sha256": checkpoint["sha256"],
            "league_policies": sorted(league["policies"]),
        }
        if "conv_calibration" in initialization:
            record["conv_ratio_after"] = initialization["conv_calibration"][
                "ratio_after"
            ]
        output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return

    key = jax.random.PRNGKey(config.seed)
    _, network_key = jax.random.split(key)
    network = build_network(config, network_key)
    trunk = network.transformer if isinstance(network, ConvCompetitionTransformer) else network
    record = {
        "architecture": config.model_architecture,
        "devices": [str(device) for device in jax.devices()],
        "transformer_trunk_sha256": _tree_sha256(trunk),
    }
    if isinstance(network, ConvCompetitionTransformer):
        observations = jax.random.normal(
            jax.random.PRNGKey(98),
            (16, config.input_channels, config.pad_to, config.pad_to),
        )
        _, calibration = calibrate_conv_token_rms(network, observations, 0.25)
        calibration = jax.device_get(calibration)
        record["conv_ratio_after"] = float(calibration["ratio_after"])
    values = jax.pmap(lambda value: value + 1)(jnp.arange(4))
    record["pmap_result"] = [int(value) for value in values]
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def run_parent(*, full_training: bool = False) -> None:
    gpu_lines = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
    if len(gpu_lines) != 8:
        raise RuntimeError(f"Expected one 8xH100 allocation, found {gpu_lines}")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        settings = {
            "transformer": (
                "0,1,2,3",
                "generals/training/configs/arch_ab_d448_8xh100_5h_transformer.toml",
            ),
            "conv": (
                "4,5,6,7",
                "generals/training/configs/arch_ab_d448_8xh100_5h_conv.toml",
            ),
        }
        workers = []
        for branch, (gpus, config) in settings.items():
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpus
            output = root / f"{branch}.json"
            workers.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "generals.training.ab_preflight",
                        "--branch-config",
                        config,
                        "--output",
                        str(output),
                        *(["--full-training"] if full_training else []),
                    ],
                    env=env,
                )
            )
        codes = [worker.wait() for worker in workers]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"Preflight branch failure: {codes}")
        transformer = json.loads((root / "transformer.json").read_text(encoding="utf-8"))
        conv = json.loads((root / "conv.json").read_text(encoding="utf-8"))
        if transformer["transformer_trunk_sha256"] != conv["transformer_trunk_sha256"]:
            raise RuntimeError("Transformer trunks do not match")
        if abs(conv["conv_ratio_after"] - 0.25) > 1e-4:
            raise RuntimeError(f"Conv calibration missed target: {conv['conv_ratio_after']}")
        print(json.dumps({"status": "passed", "transformer": transformer, "conv": conv}))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full-training", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.branch_config:
        if args.output is None:
            raise ValueError("--output is required with --branch-config")
        run_branch(args.branch_config, args.output, full_training=args.full_training)
    else:
        run_parent(full_training=args.full_training)


if __name__ == "__main__":
    main()

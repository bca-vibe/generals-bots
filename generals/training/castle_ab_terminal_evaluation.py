"""Run the common-state terminal castle audit for the λ=0.97 A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import TrainingConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _spawn(command: list[str], gpu: int, log_path: Path) -> tuple[subprocess.Popen, object]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONUNBUFFERED": "1",
            "JAX_COMPILATION_CACHE_DIR": str(
                log_path.parent / "jax_compilation_cache" / f"gpu_{gpu}"
            ),
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return process, handle


def _run_checked(commands: list[tuple[str, list[str], int]], output_dir: Path) -> None:
    workers = {}
    try:
        for name, command, gpu in commands:
            workers[name] = (*_spawn(command, gpu, output_dir / "logs" / f"{name}.log"),)
        failures = []
        for name, (process, handle) in workers.items():
            return_code = process.wait()
            handle.close()
            if return_code:
                failures.append((name, return_code))
        if failures:
            raise RuntimeError(f"Terminal castle audit failures: {failures}")
    finally:
        for process, handle in workers.values():
            if process.poll() is None:
                process.terminate()
            if not handle.closed:
                handle.close()


def run(args: argparse.Namespace) -> dict:
    control_config = TrainingConfig.from_toml(args.control_config)
    treatment_config = TrainingConfig.from_toml(args.treatment_config)
    checkpoints = {
        "control": control_config.run_dir / "terminal.eqx",
        "treatment": treatment_config.run_dir / "terminal.eqx",
    }
    for arm, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing {arm} terminal checkpoint: {checkpoint}")
    if _sha256(args.parent_checkpoint) != args.parent_sha256:
        raise ValueError("Parent checkpoint hash changed before terminal audit")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands: list[tuple[str, list[str], int]] = []
    for index, (arm, config_path) in enumerate(
        (("control", args.control_config), ("treatment", args.treatment_config))
    ):
        commands.append(
            (
                f"selfplay_{arm}",
                [
                    sys.executable,
                    "tools/evaluate_selfplay_castles.py",
                    "--checkpoint",
                    str(checkpoints[arm]),
                    "--config",
                    str(config_path),
                    "--games",
                    "4096",
                    "--batch-size",
                    "256",
                    "--seed",
                    str(args.seed),
                    "--policy",
                    "both",
                    "--sampling",
                    "categorical",
                    "--output",
                    str(args.output_dir / f"selfplay_{arm}.json"),
                ],
                4 + index,
            )
        )
        for policy_index, policy in enumerate(("raw", "ema")):
            commands.append(
                (
                    f"atlas_{arm}_{policy}",
                    [
                        sys.executable,
                        "tools/evaluate_castle_counterfactuals.py",
                        "--checkpoint",
                        str(checkpoints[arm]),
                        "--config",
                        str(config_path),
                        "--policy",
                        policy,
                        "--expected-iteration",
                        "4003",
                        "--source-checkpoint",
                        str(args.parent_checkpoint),
                        "--source-config",
                        str(args.control_config),
                        "--source-policy",
                        "ema",
                        "--source-games",
                        "1024",
                        "--collection-batch-size",
                        "512",
                        "--opportunities",
                        "2016",
                        "--repetitions",
                        "16",
                        "--pair-batch-size",
                        "16",
                        "--seed",
                        str(args.seed),
                        "--output-dir",
                        str(args.output_dir / f"atlas_{arm}_{policy}"),
                    ],
                    index * 2 + policy_index,
                )
            )
    _run_checked(commands, args.output_dir)

    reports = {}
    state_hashes = set()
    for arm in ("control", "treatment"):
        selfplay_path = args.output_dir / f"selfplay_{arm}.json"
        reports[f"selfplay_{arm}"] = json.loads(
            selfplay_path.read_text(encoding="utf-8")
        )
        for policy in ("raw", "ema"):
            atlas_dir = args.output_dir / f"atlas_{arm}_{policy}"
            atlas_report = json.loads(
                (atlas_dir / "atlas.json").read_text(encoding="utf-8")
            )
            state_hashes.add(
                atlas_report["source_sampling"]["selected_state_sha256"]
            )
            analysis_path = atlas_dir / "value_analysis.json"
            subprocess.run(
                [
                    sys.executable,
                    "tools/analyze_castle_value_probe.py",
                    "--atlas",
                    str(atlas_dir / "paired_rollouts.npz"),
                    "--selfplay",
                    str(selfplay_path),
                    "--output",
                    str(analysis_path),
                    "--seed",
                    str(args.seed + 101),
                ],
                check=True,
            )
            reports[f"atlas_{arm}_{policy}"] = atlas_report
            reports[f"value_{arm}_{policy}"] = json.loads(
                analysis_path.read_text(encoding="utf-8")
            )
    if len(state_hashes) != 1:
        raise RuntimeError(
            f"Terminal counterfactual evaluations did not use common states: {state_hashes}"
        )

    manifest = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest.append(
                {
                    "path": str(path.relative_to(args.output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    summary = {
        "status": "complete",
        "parent_checkpoint": str(args.parent_checkpoint),
        "parent_checkpoint_sha256": args.parent_sha256,
        "terminal_checkpoint_sha256": {
            arm: _sha256(path) for arm, path in checkpoints.items()
        },
        "common_selected_state_sha256": next(iter(state_hashes)),
        "reports": reports,
        "manifest": manifest,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--treatment-config", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--parent-sha256",
        default="d669c7fb28d530c5ba12e460c4e2e00b5cc5900fbdebf1da402b47e9745e8c72",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

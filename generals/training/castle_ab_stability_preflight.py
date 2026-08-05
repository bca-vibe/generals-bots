"""Run a few post-resume updates to check λ=0.97 KL stability on both arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _training_records(path: Path, first_iteration: int) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("iteration", 0) >= first_iteration and "loss" in record:
                records.append(record)
    return records


def run_branch(
    config_path: Path,
    resume: Path,
    output: Path,
    iterations: int,
) -> None:
    import jax

    from .config import TrainingConfig
    from .train import train

    production = TrainingConfig.from_toml(config_path)
    if jax.device_count() != 4:
        raise RuntimeError(f"Expected four visible GPUs, found {jax.devices()}")
    start_iteration = production.parent_final_iteration + 1
    resume_iteration = start_iteration
    run_name = f"stability_{production.run_name}"
    config = replace(
        production,
        output_dir=str(output.parent / "runs"),
        run_name=run_name,
        parent_final_iteration=resume_iteration,
        parent_final_samples=(
            resume_iteration * 2 * production.num_envs * production.num_steps * 4
        ),
        resume_checkpoint_source=str(resume),
        resume_checkpoint_sha256=_sha256(resume),
        num_iterations=resume_iteration + iterations,
        latest_checkpoint_every=0,
        checkpoint_every=iterations + 1,
        league_eval_after_training=False,
        league_eval_every=0,
        league_checkpoint_name=None,
        league_checkpoint_path=None,
        league_checkpoint_sha256=None,
        wandb_project=None,
        wandb_run_id=None,
        wandb_run_name=None,
        wandb_tags=(),
        reset_pool_every=0,
    )
    config.validate()
    train(config, resume=str(resume))
    records = _training_records(config.run_dir / "metrics.jsonl", resume_iteration + 1)
    if len(records) != iterations:
        raise RuntimeError(f"Expected {iterations} stability records, got {len(records)}")
    checked = ("loss", "approximate_kl", "gradient_norm", "explained_variance")
    if any(not math.isfinite(float(record[name])) for record in records for name in checked):
        raise RuntimeError("Nonfinite value in stability preflight")
    result = {
        "config": str(config_path),
        "resume": str(resume),
        "resume_sha256": _sha256(resume),
        "devices": [str(device) for device in jax.devices()],
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def run_parent(args: argparse.Namespace) -> None:
    settings = (
        (
            "control",
            "0,1,2,3",
            args.control_config,
            args.control_resume,
        ),
        (
            "treatment",
            "4,5,6,7",
            args.treatment_config,
            args.treatment_resume,
        ),
    )
    workers = []
    outputs = {}
    for arm, devices, config, resume in settings:
        output = args.output_dir / f"{arm}.json"
        outputs[arm] = output
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": devices,
                "PYTHONUNBUFFERED": "1",
                "JAX_COMPILATION_CACHE_DIR": str(
                    args.output_dir / "jax_compilation_cache" / arm
                ),
            }
        )
        workers.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "generals.training.castle_ab_stability_preflight",
                    "--branch-config",
                    str(config),
                    "--resume",
                    str(resume),
                    "--output",
                    str(output),
                    "--iterations",
                    str(args.iterations),
                ],
                env=environment,
            )
        )
    codes = [worker.wait() for worker in workers]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"Stability preflight failed: {codes}")
    results = {
        arm: json.loads(path.read_text(encoding="utf-8"))
        for arm, path in outputs.items()
    }
    summary = {
        "status": "passed",
        "iterations": args.iterations,
        "arms": {
            arm: {
                "iterations": [record["iteration"] for record in result["records"]],
                "kl": [record["approximate_kl"] for record in result["records"]],
                "clip_fraction": [
                    record["clip_fraction"] for record in result["records"]
                ],
                "gradient_norm": [
                    record["gradient_norm"] for record in result["records"]
                ],
            }
            for arm, result in results.items()
        },
    }
    destination = args.output_dir / "summary.json"
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument(
        "--control-config",
        type=Path,
        default=Path(
            "generals/training/configs/castle_ab_lambda097_control_from_3003.toml"
        ),
    )
    parser.add_argument(
        "--treatment-config",
        type=Path,
        default=Path(
            "generals/training/configs/castle_ab_lambda097_phi_boost_from_3003.toml"
        ),
    )
    parser.add_argument("--control-resume", type=Path)
    parser.add_argument("--treatment-resume", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/castle_ab_stability_preflight")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.branch_config:
        if args.resume is None or args.output is None:
            raise ValueError("--resume and --output are required for a branch")
        run_branch(args.branch_config, args.resume, args.output, args.iterations)
    else:
        if args.control_resume is None or args.treatment_resume is None:
            raise ValueError("Both branch resume checkpoints are required")
        run_parent(args)


if __name__ == "__main__":
    main()

"""Concurrent four-GPU-per-arm resume preflight for the castle A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _last_training_record(path: Path, iteration: int) -> dict:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("iteration") == iteration and "loss" in record:
                records.append(record)
    if len(records) != 1:
        raise RuntimeError(
            f"Expected one training record for iteration {iteration}, got {len(records)}"
        )
    return records[0]


def run_branch(
    config_path: Path,
    resume: Path,
    output: Path,
    arm: str,
) -> None:
    # Import JAX only after CUDA_VISIBLE_DEVICES is branch-specific.
    import jax

    from .config import TrainingConfig
    from .continuation_supervisor import _prepare_publication
    from .train import train

    production = TrainingConfig.from_toml(config_path)
    if jax.device_count() != 4:
        raise RuntimeError(f"{arm} expected four visible GPUs, got {jax.devices()}")
    if _sha256(resume) != production.resume_checkpoint_sha256:
        raise ValueError(f"{arm} resume checkpoint hash mismatch")
    run_name = f"preflight_{production.run_name}"
    preflight = replace(
        production,
        output_dir=str(output.parent / "runs"),
        run_name=run_name,
        num_iterations=production.parent_final_iteration + 1,
        checkpoint_every=1,
        latest_checkpoint_every=1,
        league_eval_every=1,
        league_eval_maps=8,
        league_checkpoint_maps=8,
        league_opponents=("random",),
        league_eval_policies=("raw", "ema"),
        league_eval_after_training=False,
        reset_pool_every=0,
        wandb_project=None,
        wandb_run_id=None,
        wandb_run_name=None,
        wandb_tags=(),
    )
    preflight.validate()
    train(preflight, resume=str(resume))
    iteration = production.parent_final_iteration + 1
    record = _last_training_record(preflight.run_dir / "metrics.jsonl", iteration)
    eligible = float(record["tactical_eligible_steps"])
    selected = float(record["tactical_selected_builds"])
    request_path = (
        preflight.run_dir
        / "publish_requests"
        / f"checkpoint_{iteration:06d}.json"
    )
    if not request_path.is_file():
        raise RuntimeError(f"{arm} preflight did not produce publication_request")
    publication = _prepare_publication(
        json.loads(request_path.read_text(encoding="utf-8")),
        config_path,
        preflight.run_dir,
        f"bca-vibe/generals-bot@main:runs/castle_ab_preflight/{arm}",
        preflight,
    )
    result = {
        "arm": arm,
        "devices": [str(device) for device in jax.devices()],
        "resume_sha256": _sha256(resume),
        "iteration": iteration,
        "checkpoint": json.loads(
            (preflight.run_dir / "terminal_checkpoint.json").read_text(
                encoding="utf-8"
            )
        ),
        "initialization": json.loads(
            (preflight.run_dir / "initialization.json").read_text(encoding="utf-8")
        ),
        "league": str(preflight.run_dir / f"league_{iteration:06d}.json"),
        "publication_request": str(request_path),
        "publication_manifest": str(
            preflight.run_dir
            / "publish_ready"
            / f"iteration_{iteration:06d}"
            / "manifest.json"
        ),
        "publication": publication,
        "training_metrics": record,
        "eligible_step_build_rate": selected / max(eligible, 1.0),
    }
    for required in ("league", "publication_request", "publication_manifest"):
        if not Path(result[required]).is_file():
            raise RuntimeError(f"{arm} preflight did not produce {required}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def run_parent(args: argparse.Namespace) -> None:
    gpu_lines = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
    if len(gpu_lines) != 8:
        raise RuntimeError(f"Expected one 8xH100 allocation, found {gpu_lines}")
    if _sha256(args.resume) != args.expected_sha256:
        raise ValueError("Parent resume checkpoint SHA-256 mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "control_lambda097": ("0,1,2,3", args.control_config),
        "treatment_phi_boost": ("4,5,6,7", args.treatment_config),
    }
    workers = []
    outputs = {}
    for arm, (gpus, config_path) in settings.items():
        output = args.output_dir / f"{arm}.json"
        outputs[arm] = output
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpus,
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
                    "generals.training.castle_ab_preflight",
                    "--branch-config",
                    str(config_path),
                    "--resume",
                    str(args.resume),
                    "--output",
                    str(output),
                    "--arm",
                    arm,
                ],
                env=environment,
            )
        )
    codes = [worker.wait() for worker in workers]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"Castle A/B preflight branch failure: {codes}")
    records = {
        arm: json.loads(path.read_text(encoding="utf-8"))
        for arm, path in outputs.items()
    }
    trunks = {
        record["initialization"]["transformer_trunk_sha256"]
        for record in records.values()
    }
    if len(trunks) != 1:
        raise RuntimeError(f"Control/treatment resumed different trunks: {trunks}")
    control = records["control_lambda097"]
    treatment = records["treatment_phi_boost"]
    if control["training_metrics"]["tactical_selected_builds"] != 0:
        raise RuntimeError("Control selected a tactically boosted build")
    treatment_rate = treatment["eligible_step_build_rate"]
    status = "passed"
    if not 0.001 <= treatment_rate <= 0.005:
        status = "boost_calibration_needed"
    summary = {
        "status": status,
        "source_checkpoint_sha256": args.expected_sha256,
        "shared_transformer_trunk_sha256": next(iter(trunks)),
        "target_eligible_step_build_rate": [0.001, 0.005],
        "records": records,
    }
    destination = args.output_dir / "preflight_summary.json"
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    if status != "passed":
        raise RuntimeError(
            f"Treatment boost needs calibration: eligible-step rate={treatment_rate:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-config", type=Path)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm")
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
    parser.add_argument("--output-dir", type=Path, default=Path("runs/castle_ab_preflight"))
    parser.add_argument(
        "--expected-sha256",
        default="d669c7fb28d530c5ba12e460c4e2e00b5cc5900fbdebf1da402b47e9745e8c72",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.branch_config:
        if args.output is None or not args.arm:
            raise ValueError("--output and --arm are required with --branch-config")
        run_branch(args.branch_config, args.resume, args.output, args.arm)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()

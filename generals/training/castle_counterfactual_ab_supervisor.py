"""Supervise the 4+4 GPU castle-counterfactual PPO A/B and terminal audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import TrainingConfig

EXPERIMENT_ROOT = "castle_counterfactual_ppo_ab_from_004000_20260805"
BRANCHES = {
    "control": {
        "gpus": "0,1,2,3",
        "config": "generals/training/configs/castle_counterfactual_control_from_4000.toml",
    },
    "treatment": {
        "gpus": "4,5,6,7",
        "config": "generals/training/configs/castle_counterfactual_treatment_from_4000.toml",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _terminate(workers: dict[str, subprocess.Popen]) -> None:
    for worker in workers.values():
        if worker.poll() is None:
            worker.send_signal(signal.SIGTERM)
    for worker in workers.values():
        if worker.poll() is None:
            try:
                worker.wait(timeout=20 * 60)
            except subprocess.TimeoutExpired:
                worker.kill()


def run(args: argparse.Namespace) -> None:
    actual_sha256 = _sha256(args.resume)
    if actual_sha256 != args.expected_sha256:
        raise ValueError(
            f"Resume checkpoint SHA-256 mismatch: {actual_sha256} != "
            f"{args.expected_sha256}"
        )
    gpu_lines = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
    if len(gpu_lines) != 8:
        raise RuntimeError(f"Expected one 8xH100 node, found {gpu_lines}")

    control = TrainingConfig.from_toml(BRANCHES["control"]["config"])
    treatment = TrainingConfig.from_toml(BRANCHES["treatment"]["config"])
    if control.counterfactual_castle_training:
        raise ValueError("Control unexpectedly enables counterfactual training")
    if not treatment.counterfactual_castle_training:
        raise ValueError("Treatment does not enable counterfactual training")
    if control.residual_build_kind_head:
        raise ValueError("Control unexpectedly enables the residual build-kind head")
    if not treatment.residual_build_kind_head:
        raise ValueError("Treatment does not enable the residual build-kind head")
    for name in (
        "seed",
        "parent_final_iteration",
        "num_iterations",
        "num_envs",
        "num_steps",
        "minibatch_size",
        "gae_lambda",
    ):
        if getattr(control, name) != getattr(treatment, name):
            raise ValueError(f"A/B mismatch for {name}")
    if control.num_iterations - control.parent_final_iteration != 400:
        raise ValueError("A/B is not configured for exactly 400 continuation iterations")

    workers: dict[str, subprocess.Popen] = {}

    def forward(_signum, _frame):
        _terminate(workers)

    previous = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        for arm, settings in BRANCHES.items():
            environment = os.environ.copy()
            environment.update(
                {
                    # Keep this intermediate publisher/supervisor process off
                    # the GPUs. It hands the assigned half to its train child.
                    "CUDA_VISIBLE_DEVICES": "",
                    "TRAIN_CUDA_VISIBLE_DEVICES": settings["gpus"],
                    "TRAIN_EXPECTED_JAX_DEVICE_COUNT": "4",
                    "PYTHONUNBUFFERED": "1",
                    "JAX_COMPILATION_CACHE_DIR": str(args.compilation_cache / arm),
                }
            )
            command = [
                sys.executable,
                "-u",
                "-m",
                "generals.training.continuation_supervisor",
                "--config",
                settings["config"],
                "--resume",
                str(args.resume),
                "--duration-hours",
                str(args.duration_hours),
                "--hf-root",
                f"bca-vibe/generals-bot@main:runs/{EXPERIMENT_ROOT}/{arm}",
            ]
            print(f"Starting {arm} on CUDA devices {settings['gpus']}", flush=True)
            workers[arm] = subprocess.Popen(command, env=environment)

        failed = None
        while workers:
            for arm, worker in list(workers.items()):
                return_code = worker.poll()
                if return_code is None:
                    continue
                del workers[arm]
                if return_code:
                    failed = (arm, return_code)
                    break
            if failed:
                _terminate(workers)
                raise RuntimeError(f"A/B worker failed: {failed}")
            if workers:
                time.sleep(5)

        terminal_records = {}
        for arm, settings in BRANCHES.items():
            config = TrainingConfig.from_toml(settings["config"])
            metadata = json.loads(
                (config.run_dir / "terminal_checkpoint.json").read_text()
            )
            if metadata["iteration"] != config.num_iterations:
                raise RuntimeError(
                    f"{arm} stopped at {metadata['iteration']}, not "
                    f"{config.num_iterations}"
                )
            terminal_records[arm] = metadata

        output_root = Path("runs") / EXPERIMENT_ROOT
        output_root.mkdir(parents=True, exist_ok=True)
        round_robin = output_root / "round_robin.json"
        milestones = [
            control.parent_final_iteration + offset
            for offset in (100, 200, 300, 400)
        ]
        evaluation_environment = os.environ.copy()
        evaluation_environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                "EXPECTED_JAX_DEVICE_COUNT": "8",
                "JAX_COMPILATION_CACHE_DIR": str(args.compilation_cache / "evaluation"),
            }
        )
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "generals.training.evaluate_checkpoint_round_robin",
                "--control-config",
                BRANCHES["control"]["config"],
                "--treatment-config",
                BRANCHES["treatment"]["config"],
                "--parent-checkpoint",
                str(args.resume),
                "--output",
                str(round_robin),
                "--maps",
                str(args.round_robin_maps),
                "--iterations",
                *(str(value) for value in milestones),
                "--policies",
                "raw",
            ],
            check=True,
            env=evaluation_environment,
        )
        terminal_audit = output_root / "terminal_castle_audit"
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "generals.training.castle_ab_terminal_evaluation",
                "--control-config",
                BRANCHES["control"]["config"],
                "--treatment-config",
                BRANCHES["treatment"]["config"],
                "--parent-checkpoint",
                str(args.resume),
                "--parent-sha256",
                args.expected_sha256,
                "--output-dir",
                str(terminal_audit),
            ],
            check=True,
            env=evaluation_environment,
        )
        summary = {
            "status": "complete",
            "source_checkpoint_sha256": actual_sha256,
            "terminal_checkpoints": terminal_records,
            "round_robin": str(round_robin),
            "terminal_castle_audit": str(terminal_audit / "manifest.json"),
        }
        (output_root / "supervisor_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        _terminate(workers)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument(
        "--expected-sha256",
        default="7c553a864a3a7e7b751b1e3e4695befdae0cb099d898ec770215399e41dac311",
    )
    parser.add_argument(
        "--compilation-cache",
        type=Path,
        default=Path("/home/dev/.cache/castle_counterfactual_ab"),
    )
    parser.add_argument("--duration-hours", type=float, default=4.0)
    parser.add_argument("--round-robin-maps", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

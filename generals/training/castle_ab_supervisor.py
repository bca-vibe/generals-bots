"""Supervise the fixed-iteration castle λ=0.97 control/treatment A/B."""

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

EXPERIMENT_ROOT = "castle_phi_boost_ab_lambda097_from_003003_20260804"
BRANCHES = {
    "control_lambda097": {
        "gpus": "0,1,2,3",
        "config": "generals/training/configs/castle_ab_lambda097_control_from_3003.toml",
    },
    "treatment_phi_boost": {
        "gpus": "4,5,6,7",
        "config": "generals/training/configs/castle_ab_lambda097_phi_boost_from_3003.toml",
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
            f"Resume checkpoint SHA-256 mismatch: {actual_sha256} != {args.expected_sha256}"
        )
    gpu_lines = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
    if len(gpu_lines) != 8:
        raise RuntimeError(f"Expected one 8xH100 allocation, found {gpu_lines}")

    workers: dict[str, subprocess.Popen] = {}

    def forward(signum, _frame):
        _terminate(workers)

    previous_handlers = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        for arm, settings in BRANCHES.items():
            config = TrainingConfig.from_toml(settings["config"])
            if config.gae_lambda != 0.97:
                raise ValueError(f"{arm} is not explicitly configured with lambda=0.97")
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": settings["gpus"],
                    "PYTHONUNBUFFERED": "1",
                    "JAX_COMPILATION_CACHE_DIR": str(
                        args.compilation_cache / arm
                    ),
                }
            )
            hf_root = (
                f"bca-vibe/generals-bot@main:runs/{EXPERIMENT_ROOT}/{arm}"
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
                "24",
                "--hf-root",
                hf_root,
            ]
            print(
                f"Starting {arm} on CUDA devices {settings['gpus']} with shared lambda=0.97",
                flush=True,
            )
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
            metadata_path = config.run_dir / "terminal_checkpoint.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["iteration"] != 4003:
                raise RuntimeError(f"{arm} stopped at {metadata['iteration']}, not 4003")
            terminal_records[arm] = metadata

        round_robin_output = Path("runs") / EXPERIMENT_ROOT / "round_robin.json"
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "generals.training.evaluate_checkpoint_round_robin",
                "--control-config",
                BRANCHES["control_lambda097"]["config"],
                "--treatment-config",
                BRANCHES["treatment_phi_boost"]["config"],
                "--parent-checkpoint",
                str(args.resume),
                "--output",
                str(round_robin_output),
                "--maps",
                str(args.round_robin_maps),
                "--policies",
                "raw",
                "ema",
            ],
            check=True,
        )
        terminal_audit_dir = (
            Path("runs") / EXPERIMENT_ROOT / "terminal_castle_audit"
        )
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "generals.training.castle_ab_terminal_evaluation",
                "--control-config",
                BRANCHES["control_lambda097"]["config"],
                "--treatment-config",
                BRANCHES["treatment_phi_boost"]["config"],
                "--parent-checkpoint",
                str(args.resume),
                "--parent-sha256",
                args.expected_sha256,
                "--output-dir",
                str(terminal_audit_dir),
            ],
            check=True,
        )
        summary = {
            "status": "complete",
            "source_checkpoint_sha256": actual_sha256,
            "terminal_checkpoints": terminal_records,
            "round_robin": str(round_robin_output),
            "terminal_castle_audit": str(terminal_audit_dir / "manifest.json"),
        }
        summary_path = Path("runs") / EXPERIMENT_ROOT / "supervisor_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        _terminate(workers)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument(
        "--expected-sha256",
        default="d669c7fb28d530c5ba12e460c4e2e00b5cc5900fbdebf1da402b47e9745e8c72",
    )
    parser.add_argument(
        "--compilation-cache",
        type=Path,
        default=Path("/home/dev/.cache/castle_ab_lambda097"),
    )
    parser.add_argument("--round-robin-maps", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

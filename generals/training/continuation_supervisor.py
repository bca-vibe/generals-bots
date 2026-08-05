"""Run a timed continuation and prepare checkpoint publications off the PPO path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from .config import TrainingConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _append_status(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _prepare_publication(
    request: dict, config_path: Path, run_dir: Path, hf_root: str
) -> dict:
    iteration = int(request["iteration"])
    checkpoint = Path(request["checkpoint_path"])
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    actual_sha256 = _sha256(checkpoint)
    if actual_sha256 != request["checkpoint_sha256"]:
        raise ValueError(
            f"checkpoint {iteration} changed after handoff: "
            f"{actual_sha256} != {request['checkpoint_sha256']}"
        )

    final_dir = run_dir / "publish_ready" / f"iteration_{iteration:06d}"
    temporary = final_dir.with_name(f".{final_dir.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    full_destination = temporary / "training_checkpoint.eqx"
    _hardlink_or_copy(checkpoint, full_destination)
    source = f"{hf_root}/checkpoints/iteration_{iteration:06d}/training_checkpoint.eqx"
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    bundles = {}
    for policy in ("ema", "raw"):
        competition_dir = temporary / f"competition_{policy}"
        subprocess.run(
            [
                sys.executable,
                "tools/export_competition_checkpoint.py",
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(competition_dir),
                "--expected-sha256",
                actual_sha256,
                "--source",
                source,
                "--policy",
                policy,
            ],
            check=True,
            env=environment,
        )
        bundle = temporary / f"competition_{policy}.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as archive:
            for file_path in sorted(competition_dir.iterdir()):
                if file_path.is_file():
                    archive.write(file_path, arcname=file_path.name)
        bundles[policy] = bundle

    ema_bundle = bundles["ema"]
    raw_bundle = bundles["raw"]

    record = {
        **request,
        "event": "bundle_ready",
        "checkpoint_path": str(final_dir / "training_checkpoint.eqx"),
        "checkpoint_sha256": actual_sha256,
        "checkpoint_bytes": full_destination.stat().st_size,
        "competition_path": str(final_dir / "competition_ema.zip"),
        "competition_sha256": _sha256(ema_bundle),
        "competition_bytes": ema_bundle.stat().st_size,
        "competition_bundle_available": True,
        "competition_raw_path": str(final_dir / "competition_raw.zip"),
        "competition_raw_sha256": _sha256(raw_bundle),
        "competition_raw_bytes": raw_bundle.stat().st_size,
        "raw_competition_bundle_available": True,
        "remote_path": f"{hf_root}/checkpoints/iteration_{iteration:06d}",
        "remote_checkpoint_path": (
            f"{hf_root}/checkpoints/iteration_{iteration:06d}/"
            "training_checkpoint.eqx"
        ),
        "remote_competition_path": (
            f"{hf_root}/checkpoints/iteration_{iteration:06d}/competition_ema.zip"
        ),
        "remote_raw_competition_path": (
            f"{hf_root}/checkpoints/iteration_{iteration:06d}/competition_raw.zip"
        ),
    }
    _write_json_atomic(temporary / "manifest.json", record)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(temporary, final_dir)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--duration-hours", type=float, required=True)
    parser.add_argument(
        "--hf-root",
        default="bca-vibe/generals-bot@main:runs/conv_d448_8xh100_12h_cont_from_001313_20260803",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = TrainingConfig.from_toml(config_path)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "publish_status.jsonl"
    command = [
        sys.executable,
        "-u",
        "-m",
        "generals.training.train",
        "--config",
        str(config_path),
        "--resume",
        args.resume,
        "--duration-hours",
        str(args.duration_hours),
    ]
    child = subprocess.Popen(command)
    attempted: set[tuple[int, str]] = set()

    def forward(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    previous = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        while True:
            requests = sorted((run_dir / "publish_requests").glob("checkpoint_*.json"))
            for request_path in requests:
                request = json.loads(request_path.read_text(encoding="utf-8"))
                identity = (int(request["iteration"]), request["checkpoint_sha256"])
                if identity in attempted:
                    continue
                attempted.add(identity)
                started = time.perf_counter()
                try:
                    record = _prepare_publication(
                        request, config_path, run_dir, args.hf_root
                    )
                    record["bundle_prepare_seconds"] = time.perf_counter() - started
                    _append_status(status_path, record)
                    print(
                        f"Prepared asynchronous checkpoint publication for "
                        f"iteration {record['iteration']}"
                    )
                except Exception as error:  # noqa: BLE001 - publisher cannot stop PPO.
                    _append_status(
                        status_path,
                        {
                            **request,
                            "event": "bundle_failed",
                            "error": f"{type(error).__name__}: {error}",
                        },
                    )
                    print(
                        f"Checkpoint publisher failed at iteration "
                        f"{request['iteration']}: {error}",
                        file=sys.stderr,
                    )
            return_code = child.poll()
            if return_code is not None:
                remaining = [
                    path
                    for path in (run_dir / "publish_requests").glob("checkpoint_*.json")
                    if (
                        int(json.loads(path.read_text())["iteration"]),
                        json.loads(path.read_text())["checkpoint_sha256"],
                    )
                    not in attempted
                ]
                if not remaining:
                    if return_code:
                        raise SystemExit(return_code)
                    return
            time.sleep(2)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()

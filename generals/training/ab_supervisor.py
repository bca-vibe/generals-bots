"""Supervise the independent four-GPU architecture A/B batch experiment."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

EXPERIMENT_ID = "arch_ab_d448_8xh100_5h_retry1_20260803"
TOTAL_BUDGET_SECONDS = 5 * 60 * 60
FINALIZATION_RESERVE_SECONDS = 35 * 60
CAPTURE_FLOOR_SECONDS = 5 * 60
POLL_SECONDS = 5
FIRST_ITERATION_TIMEOUT_SECONDS = 20 * 60

BRANCHES = {
    "transformer": {
        "gpus": "0,1,2,3",
        "config": "generals/training/configs/arch_ab_d448_8xh100_5h_transformer.toml",
        "run_name": f"{EXPERIMENT_ID}_transformer",
    },
    "conv": {
        "gpus": "4,5,6,7",
        "config": "generals/training/configs/arch_ab_d448_8xh100_5h_conv.toml",
        "run_name": f"{EXPERIMENT_ID}_conv",
    },
}


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def copy_matching(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    for path in source.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        atomic_copy(path, destination / path.relative_to(source))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_request(path: str, *, method: str = "GET") -> dict | None:
    base = os.environ.get("GMN_METADATA_URL")
    if not base:
        return None
    request = urllib.request.Request(
        f"{base}{path}",
        method=method,
        headers={"Metadata-Flavor": "givemeanode"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code in {429, 503}:
            return {"status": "retry", "code": error.code}
        raise


def checkpoint_timestamp() -> str | None:
    response = metadata_request("/v1/job") or {}
    return response.get("checkpointed_at")


def capture_checkpoint(previous_timestamp: str | None) -> str | None:
    # A final capture may arrive just after a periodic one. Allow the full
    # five-minute rate floor plus commit time rather than failing finalization.
    deadline = time.monotonic() + 420
    while time.monotonic() < deadline:
        response = metadata_request("/v1/checkpoint", method="POST")
        if response is None:
            return previous_timestamp
        if response.get("status") == "retry":
            time.sleep(5)
            continue
        while time.monotonic() < deadline:
            current = checkpoint_timestamp()
            if current and current != previous_timestamp:
                print(f"givemeanode checkpoint committed at {current}", flush=True)
                return current
            time.sleep(2)
    raise TimeoutError("givemeanode checkpoint capture did not commit within 420 seconds")


def wandb_stop_requested() -> bool:
    """A `stop-requested` W&B tag is the cooperative remote-stop control."""
    try:
        import wandb

        entity = os.environ.get("WANDB_ENTITY", "bcarnold-independent")
        api = wandb.Api(timeout=20)
        for run_id in (
            "arch-ab-d448-5h-retry1-transformer-20260803",
            "arch-ab-d448-5h-retry1-conv-20260803",
        ):
            run = api.run(f"{entity}/generals-bots/{run_id}")
            if "stop-requested" in run.tags:
                return True
    except Exception as error:  # noqa: BLE001 - control polling cannot kill training.
        print(f"W&B stop-control poll failed harmlessly: {error}", flush=True)
    return False


def branch_run_dir(branch: str) -> Path:
    return Path("runs") / BRANCHES[branch]["run_name"]


def branch_has_training_iteration(branch: str) -> bool:
    metrics_path = branch_run_dir(branch) / "metrics.jsonl"
    if not metrics_path.is_file():
        return False
    with metrics_path.open(encoding="utf-8") as handle:
        return any('"loss"' in line and '"iteration": 1' in line for line in handle)


def sync_branch(branch: str, checkpoint_root: Path, output_root: Path) -> None:
    source = branch_run_dir(branch)
    if not source.exists():
        return
    output_branch = output_root / "branches" / branch
    copy_matching(source, output_branch)

    recovery_branch = checkpoint_root / "branches" / branch
    for filename in (
        "latest.eqx",
        "latest_checkpoint.json",
        "terminal.eqx",
        "terminal_checkpoint.json",
        "metrics.jsonl",
        "config.json",
        "initialization.json",
        "conv_calibration.json",
    ):
        path = source / filename
        if path.is_file():
            atomic_copy(path, recovery_branch / filename)
    for path in source.glob("checkpoint_*.eqx"):
        atomic_copy(path, recovery_branch / "archive" / path.name)
    for path in source.glob("league_*.json"):
        atomic_copy(path, recovery_branch / "league" / path.name)


def restore_branches(checkpoint_root: Path) -> dict[str, Path]:
    resumes: dict[str, Path] = {}
    for branch in BRANCHES:
        saved = checkpoint_root / "branches" / branch
        if saved.exists():
            copy_matching(saved, branch_run_dir(branch))
        latest = branch_run_dir(branch) / "latest.eqx"
        if latest.is_file():
            resumes[branch] = latest
    return resumes


def start_workers(
    resumes: dict[str, Path],
    soft_deadline: float,
    gate_path: Path,
    checkpoint_root: Path,
) -> dict[str, subprocess.Popen]:
    workers = {}
    for branch, settings in BRANCHES.items():
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = settings["gpus"]
        env["PYTHONUNBUFFERED"] = "1"
        env["JAX_COMPILATION_CACHE_DIR"] = str(
            checkpoint_root / "jax_compilation_cache" / branch
        )
        command = [
            sys.executable,
            "-m",
            "generals.training.train",
            "--config",
            settings["config"],
            "--stop-at-unix",
            str(soft_deadline),
        ]
        if branch in resumes:
            command.extend(["--resume", str(resumes[branch])])
        else:
            command.extend(["--initialization-gate", str(gate_path)])
        print(f"Starting {branch} on CUDA devices {settings['gpus']}", flush=True)
        workers[branch] = subprocess.Popen(command, env=env)
    return workers


def approve_initialization(gate_path: Path) -> None:
    records = {}
    deadline = time.monotonic() + 20 * 60
    while time.monotonic() < deadline:
        for branch in BRANCHES:
            path = branch_run_dir(branch) / "initialization.json"
            if branch not in records and path.is_file():
                records[branch] = json.loads(path.read_text(encoding="utf-8"))
        if len(records) == len(BRANCHES):
            break
        time.sleep(1)
    if len(records) != len(BRANCHES):
        raise TimeoutError("Both initialization records did not appear within 20 minutes")
    transformer_hash = records["transformer"]["transformer_trunk_sha256"]
    conv_hash = records["conv"]["transformer_trunk_sha256"]
    if transformer_hash != conv_hash:
        raise RuntimeError(
            f"Transformer trunk mismatch: transformer={transformer_hash}, conv={conv_hash}"
        )
    ratio = records["conv"]["conv_calibration"]["ratio_after"]
    if abs(ratio - 0.25) > 1e-4:
        raise RuntimeError(f"Conv calibration ratio {ratio} is not 0.25")
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(
        json.dumps(
            {"transformer_trunk_sha256": transformer_hash, "conv_ratio_after": ratio},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Initialization approved: trunk={transformer_hash}, conv ratio={ratio}", flush=True)


def terminate_workers(workers: dict[str, subprocess.Popen]) -> None:
    for worker in workers.values():
        if worker.poll() is None:
            worker.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + FINALIZATION_RESERVE_SECONDS
    for worker in workers.values():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            worker.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            worker.kill()


def write_manifest(root: Path) -> Path:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256.json":
            entries.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    destination = root / "manifest.sha256.json"
    destination.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def run_original_baselines(output_root: Path, hard_deadline: float) -> int:
    if hard_deadline - time.time() < 5 * 60:
        print("Deferring original baseline league: less than five minutes remain", flush=True)
        return 0
    checkpoints = [
        Path("checkpoints/smoke_8xh100/checkpoint_000540.eqx"),
        Path("checkpoints/smoke_8xh100/checkpoint_000880.eqx"),
        Path("checkpoints/smoke_8xh100/checkpoint_001260.eqx"),
    ]
    command = [
        sys.executable,
        "-m",
        "generals.training.evaluate_checkpoints",
        "--config",
        "runs/smoke_8xh100/smoke_8xh100.toml",
        "--output-dir",
        str(output_root / "original_baselines"),
        *(str(path) for path in checkpoints),
    ]
    return subprocess.run(command, check=False).returncode


def main() -> int:
    checkpoint_root = Path(os.environ.get("GMN_CHECKPOINT_DIR", "/tmp/gmn-checkpoint"))
    output_root = Path(os.environ.get("GMN_OUTPUT_DIR", "/tmp/gmn-output")) / EXPERIMENT_ID
    result_path = Path(os.environ.get("GMN_RESULT_PATH", "/tmp/gmn-result.json"))
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    state_path = checkpoint_root / "supervisor_state.json"
    prior_state = (
        json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    )
    used_before_attempt = float(prior_state.get("used_seconds", 0.0))
    attempt_started = time.monotonic()
    remaining = max(0.0, TOTAL_BUDGET_SECONDS - used_before_attempt)
    hard_deadline = time.time() + remaining
    soft_deadline = max(time.time(), hard_deadline - FINALIZATION_RESERVE_SECONDS)
    print(
        f"Experiment budget: used={used_before_attempt:.1f}s, remaining={remaining:.1f}s, "
        f"soft deadline={soft_deadline:.3f}",
        flush=True,
    )

    resumes = restore_branches(checkpoint_root)
    gate_path = Path("runs") / EXPERIMENT_ID / "initialization_approved.json"
    dmon_path = output_root / "system" / "nvidia_smi_dmon.log"
    dmon_path.parent.mkdir(parents=True, exist_ok=True)
    dmon_handle = dmon_path.open("w", encoding="utf-8")
    dmon = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "pucvmet", "-d", "1"],
        stdout=dmon_handle,
        stderr=subprocess.STDOUT,
    )
    workers = start_workers(resumes, soft_deadline, gate_path, checkpoint_root)
    workers_started_at = time.monotonic()
    try:
        if len(resumes) != len(BRANCHES):
            approve_initialization(gate_path)

        previous_latest = {branch: 0 for branch in BRANCHES}
        last_capture = 0.0
        last_control_poll = 0.0
        early_stop_requested = False
        checkpointed_at = checkpoint_timestamp()
        while any(worker.poll() is None for worker in workers.values()):
            failed = {
                branch: worker.returncode
                for branch, worker in workers.items()
                if worker.poll() not in {None, 0}
            }
            if failed:
                raise RuntimeError(f"Training worker failed: {failed}")
            changed = False
            for branch in BRANCHES:
                latest = branch_run_dir(branch) / "latest.eqx"
                if latest.is_file() and latest.stat().st_mtime_ns > previous_latest[branch]:
                    previous_latest[branch] = latest.stat().st_mtime_ns
                    changed = True
            now = time.monotonic()
            if now - workers_started_at >= FIRST_ITERATION_TIMEOUT_SECONDS:
                pending = [
                    branch
                    for branch in BRANCHES
                    if not branch_has_training_iteration(branch)
                ]
                if pending:
                    raise TimeoutError(
                        "Branches did not finish iteration 1 within "
                        f"{FIRST_ITERATION_TIMEOUT_SECONDS}s: {pending}"
                    )
            if now - last_control_poll >= 60:
                last_control_poll = now
                if wandb_stop_requested():
                    early_stop_requested = True
                    print("Cooperative early stop requested through W&B", flush=True)
                    for worker in workers.values():
                        if worker.poll() is None:
                            worker.send_signal(signal.SIGTERM)
            if changed and now - last_capture >= CAPTURE_FLOOR_SECONDS:
                for branch in BRANCHES:
                    sync_branch(branch, checkpoint_root, output_root)
                state = {
                    "experiment_id": EXPERIMENT_ID,
                    "used_seconds": used_before_attempt + now - attempt_started,
                    "captured_at_unix": time.time(),
                }
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                checkpointed_at = capture_checkpoint(checkpointed_at)
                last_capture = time.monotonic()
            time.sleep(POLL_SECONDS)

        return_codes = {branch: worker.wait() for branch, worker in workers.items()}
        if any(code != 0 for code in return_codes.values()):
            raise RuntimeError(f"Training workers exited unsuccessfully: {return_codes}")
    except BaseException:
        terminate_workers(workers)
        for branch in BRANCHES:
            sync_branch(branch, checkpoint_root, output_root)
        write_manifest(output_root)
        dmon.terminate()
        dmon.wait(timeout=10)
        dmon_handle.close()
        raise

    dmon.terminate()
    try:
        dmon.wait(timeout=10)
    except subprocess.TimeoutExpired:
        dmon.kill()
    dmon_handle.close()

    for branch in BRANCHES:
        sync_branch(branch, checkpoint_root, output_root)
    state_path.write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "used_seconds": used_before_attempt + time.monotonic() - attempt_started,
                "completed_training": True,
                "early_stop_requested": early_stop_requested,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    checkpointed_at = capture_checkpoint(checkpointed_at)

    baseline_code = run_original_baselines(output_root, hard_deadline)
    manifest_path = write_manifest(output_root)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete" if baseline_code == 0 else "complete_baseline_eval_failed",
        "transformer_terminal": json.loads(
            (branch_run_dir("transformer") / "terminal_checkpoint.json").read_text()
        ),
        "conv_terminal": json.loads(
            (branch_run_dir("conv") / "terminal_checkpoint.json").read_text()
        ),
        "checkpointed_at": checkpointed_at,
        "manifest_sha256": sha256(manifest_path),
        "baseline_exit_code": baseline_code,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    # Training and its terminal artifacts are complete even if the ancillary
    # historical-baseline evaluation needs to be retried from those outputs.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

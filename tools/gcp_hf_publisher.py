"""Upload prepared continuation checkpoints to Hugging Face and verify them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--path-prefix", required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def _remote_checkpoint(api: HfApi, repo_id: str, revision: str, path: str):
    matches = api.get_paths_info(
        repo_id=repo_id,
        paths=[path],
        revision=revision,
        repo_type="model",
        expand=True,
    )
    return matches[0] if matches else None


def _publish_one(
    api: HfApi,
    directory: Path,
    repo_id: str,
    revision: str,
    path_prefix: str,
) -> None:
    marker = directory / ".hf_export_complete"
    if marker.exists():
        return
    manifest_path = directory / "manifest.json"
    checkpoint = directory / "training_checkpoint.eqx"
    if not manifest_path.is_file() or not checkpoint.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha256 = manifest["checkpoint_sha256"]
    actual_sha256 = _sha256(checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Local checkpoint hash mismatch in {directory}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    remote_dir = f"{path_prefix.rstrip('/')}/checkpoints/{directory.name}"
    api.upload_folder(
        repo_id=repo_id,
        folder_path=directory,
        path_in_repo=remote_dir,
        revision=revision,
        repo_type="model",
        commit_message=f"Upload continuation checkpoint {directory.name}",
        ignore_patterns=[".hf_export_complete"],
    )
    remote_path = f"{remote_dir}/training_checkpoint.eqx"
    remote = _remote_checkpoint(api, repo_id, revision, remote_path)
    if remote is None or remote.size != checkpoint.stat().st_size:
        raise RuntimeError(f"Remote size verification failed for {remote_path}")
    remote_sha256 = None
    remote_lfs = getattr(remote, "lfs", None)
    if isinstance(remote_lfs, dict):
        remote_sha256 = remote_lfs.get("sha256")
    elif remote_lfs is not None:
        remote_sha256 = getattr(remote_lfs, "sha256", None)
    if remote_sha256 is not None and remote_sha256 != actual_sha256:
        raise RuntimeError(
            f"Remote hash verification failed for {remote_path}: "
            f"{remote_sha256} != {actual_sha256}"
        )
    marker.write_text(
        json.dumps(
            {
                "checkpoint_sha256": actual_sha256,
                "completed_at_unix": time.time(),
                "remote_path": remote_path,
                "remote_size": remote.size,
                "remote_sha256": remote_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)
    idle_after_stop = 0
    while True:
        pending = []
        ready_root = args.run_dir / "publish_ready"
        if ready_root.is_dir():
            pending = [
                directory
                for directory in sorted(ready_root.glob("iteration_*"))
                if directory.is_dir()
                and not (directory / ".hf_export_complete").exists()
            ]
        for directory in pending:
            _publish_one(
                api,
                directory,
                args.repo_id,
                args.revision,
                args.path_prefix,
            )
        if args.stop_file.exists():
            idle_after_stop = idle_after_stop + 1 if not pending else 0
            if idle_after_stop >= 2:
                return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

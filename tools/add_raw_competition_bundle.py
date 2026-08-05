"""Add a competition-ready raw-policy ZIP to an existing publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--publication-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    publication_dir = Path(args.publication_dir)
    manifest_path = publication_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = publication_dir / "training_checkpoint.eqx"
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != manifest["checkpoint_sha256"]:
        raise ValueError("publication checkpoint hash does not match its manifest")

    final_bundle = publication_dir / "competition_raw.zip"
    with tempfile.TemporaryDirectory(
        prefix=".competition_raw.", dir=publication_dir
    ) as temporary_name:
        temporary = Path(temporary_name)
        output_dir = temporary / "competition_raw"
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "JAX_PLATFORMS": "cpu",
                "XLA_FLAGS": (
                    "--xla_cpu_multi_thread_eigen=false "
                    "intra_op_parallelism_threads=1"
                ),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )
        subprocess.run(
            [
                sys.executable,
                "tools/export_competition_checkpoint.py",
                "--config",
                args.config,
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--expected-sha256",
                checkpoint_sha256,
                "--source",
                manifest["remote_checkpoint_path"],
                "--policy",
                "raw",
            ],
            check=True,
            env=environment,
        )
        temporary_bundle = temporary / "competition_raw.zip"
        with zipfile.ZipFile(
            temporary_bundle, "w", compression=zipfile.ZIP_STORED
        ) as archive:
            for file_path in sorted(output_dir.iterdir()):
                if file_path.is_file():
                    archive.write(file_path, arcname=file_path.name)
        shutil.move(temporary_bundle, final_bundle)

    remote_path = manifest["remote_path"]
    manifest.update(
        {
            "competition_raw_path": str(final_bundle),
            "competition_raw_sha256": _sha256(final_bundle),
            "competition_raw_bytes": final_bundle.stat().st_size,
            "raw_competition_bundle_available": True,
            "remote_raw_competition_path": f"{remote_path}/competition_raw.zip",
        }
    )
    temporary_manifest = manifest_path.with_name(".manifest.json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

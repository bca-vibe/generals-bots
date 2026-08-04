"""Append a verified remote checkpoint-export result to its durable status log."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--upload-seconds", type=float, required=True)
    parser.add_argument("--remote-hash-verified", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    record.update(
        {
            "event": "upload_complete",
            "complete": True,
            "hash_verified": args.remote_hash_verified,
            "upload_seconds": args.upload_seconds,
        }
    )
    status_file = Path(args.status_file)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with status_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == "__main__":
    main()

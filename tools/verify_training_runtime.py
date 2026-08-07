#!/usr/bin/env python3
"""Fail fast when a training node is using an unintended Python/JAX runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="3.11", help="Expected Python major.minor")
    parser.add_argument("--jax", default="0.10.2", help="Expected JAX version")
    parser.add_argument("--jaxlib", default="0.10.2", help="Expected jaxlib version")
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Expected JAX device count; omit to avoid initializing devices",
    )
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Required package is not installed: {name}") from exc


def main() -> None:
    args = parse_args()
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    versions = {
        "python": actual_python,
        "python_full": platform.python_version(),
        "jax": package_version("jax"),
        "jaxlib": package_version("jaxlib"),
        "jax_cuda12_plugin": package_version("jax-cuda12-plugin"),
        "jax_cuda12_pjrt": package_version("jax-cuda12-pjrt"),
    }
    expected = {
        "python": args.python,
        "jax": args.jax,
        "jaxlib": args.jaxlib,
        "jax_cuda12_plugin": args.jax,
        "jax_cuda12_pjrt": args.jax,
    }
    mismatches = {
        name: {"expected": expected[name], "actual": actual}
        for name, actual in versions.items()
        if name in expected and actual != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"Training runtime mismatch: {json.dumps(mismatches, sort_keys=True)}")

    if args.devices is not None:
        import jax

        devices = [str(device) for device in jax.devices()]
        if len(devices) != args.devices:
            raise RuntimeError(
                f"Expected {args.devices} JAX devices, got {len(devices)}: {devices}"
            )
        versions["devices"] = devices

    print(json.dumps(versions, sort_keys=True))


if __name__ == "__main__":
    main()

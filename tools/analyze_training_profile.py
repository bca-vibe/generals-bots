#!/usr/bin/env python3
"""Summarize smoke-run throughput, phase timings, and sampled GPU telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def read_metrics(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def read_gpu_samples(path: Path) -> list[dict[str, float]]:
    samples = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                samples.append(
                    {
                        name: float(value)
                        for name, value in row.items()
                        if name != "timestamp" and value not in (None, "", "[N/A]")
                    }
                )
            except ValueError:
                continue
    return samples


def summarize(args) -> dict:
    main_records = read_metrics(args.main_metrics)
    main_training = [
        record for record in main_records if "samples_per_second" in record
    ]
    steady = [
        record
        for record in main_records
        if "samples_per_second" in record and int(record["iteration"]) > 5
    ]
    if not steady:
        raise ValueError(
            "main metrics contain no steady-state iterations after iteration 5"
        )
    phase_records = [
        record
        for record in read_metrics(args.phase_metrics)
        if "rollout_seconds" in record and int(record["iteration"]) > 5
    ]
    if not phase_records:
        raise ValueError("phase metrics contain no timed iterations after iteration 5")
    all_phase_training = [
        record
        for record in read_metrics(args.phase_metrics)
        if "samples_per_second" in record
    ]

    phase_medians = {
        name: statistics.median([float(record[name]) for record in phase_records])
        for name in (
            "iteration_seconds",
            "rollout_seconds",
            "update_seconds",
            "host_seconds",
        )
    }
    iteration = phase_medians["iteration_seconds"]
    phase_shares = {
        name.removesuffix("_seconds"): value / iteration
        for name, value in phase_medians.items()
        if name != "iteration_seconds"
    }

    gpu_samples = read_gpu_samples(args.gpu_csv)
    gpu = {}
    for name in (
        "gpu_util_pct",
        "memory_util_pct",
        "memory_used_mib",
        "memory_total_mib",
        "power_w",
        "sm_clock_mhz",
    ):
        values = [sample[name] for sample in gpu_samples if name in sample]
        if values:
            gpu[name] = distribution(values)
    if "memory_used_mib" in gpu and "memory_total_mib" in gpu:
        gpu["peak_memory_fraction"] = (
            gpu["memory_used_mib"]["maximum"] / gpu["memory_total_mib"]["maximum"]
        )

    host_share = phase_shares["host"]
    median_gpu_util = gpu.get("gpu_util_pct", {}).get("median")
    recommendations = []
    if host_share >= 0.05:
        recommendations.append(
            {
                "classification": "promising-but-unvalidated",
                "area": "host dispatch and bookkeeping",
                "evidence": f"{host_share:.1%} of synchronized iteration time is outside rollout/update",
                "upper_bound_speedup": host_share,
                "next_test": "A same-seed, bit-identical A/B that reduces dispatches or overlaps pool generation",
            }
        )
    else:
        recommendations.append(
            {
                "classification": "not-worthwhile",
                "area": "host dispatch and bookkeeping",
                "evidence": f"Only {host_share:.1%} of synchronized iteration time is outside rollout/update",
                "upper_bound_speedup": host_share,
            }
        )
    if median_gpu_util is not None and median_gpu_util < 85.0:
        recommendations.append(
            {
                "classification": "promising-but-unvalidated",
                "area": "device launch gaps or undersized kernels",
                "evidence": f"Median sampled GPU utilization is {median_gpu_util:.1f}%",
                "next_test": "Inspect the XPlane trace before changing any batch or model math",
            }
        )
    else:
        recommendations.append(
            {
                "classification": "not-worthwhile",
                "area": "single-GPU occupancy",
                "evidence": (
                    "Sampled GPU utilization is already high"
                    if median_gpu_util is not None
                    else "GPU telemetry unavailable"
                ),
            }
        )
    recommendations.append(
        {
            "classification": "measured-safe",
            "area": "persistent JAX compilation cache",
            "evidence": (
                "The diagnostic passes reuse the production compilation cache without changing the training program"
            ),
            "next_test": "Retain a per-shape cache on production nodes and record cold/warm startup separately",
        }
    )

    cache_savings_seconds = max(
        0.0,
        float(main_training[0]["iteration_seconds"])
        - float(all_phase_training[0]["iteration_seconds"]),
    )
    event_timings = {
        key.removeprefix("performance/"): float(value)
        for record in main_records
        for key, value in record.items()
        if key.startswith("performance/")
    }

    trace_status = "not requested"
    if args.trace_status and args.trace_status.exists():
        trace_status = args.trace_status.read_text(encoding="utf-8").strip()

    cprofile_excerpt = ""
    if args.cprofile_text and args.cprofile_text.exists():
        cprofile_excerpt = "\n".join(
            args.cprofile_text.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()[:45]
        )

    return {
        "steady_state_iterations": len(steady),
        "samples_per_second": distribution(
            [float(record["samples_per_second"]) for record in steady]
        ),
        "iteration_seconds": distribution(
            [float(record["iteration_seconds"]) for record in steady]
        ),
        "phase_medians_seconds": phase_medians,
        "phase_shares": phase_shares,
        "event_timings_seconds": event_timings,
        "compilation_cache_first_iteration_savings_seconds": cache_savings_seconds,
        "gpu_samples": len(gpu_samples),
        "gpu": gpu,
        "recommendations": recommendations,
        "cprofile_excerpt": cprofile_excerpt,
        "trace_status": trace_status,
        "limitations": [
            "One H100 cannot characterize NCCL or multi-device pmap scaling.",
            "nvidia-smi samples are coarse and cannot identify individual kernel launch gaps.",
            "No change to model math, sample order, precision, optimizer order, or curriculum was benchmarked.",
        ],
    }


def render_markdown(summary: dict) -> str:
    throughput = summary["samples_per_second"]
    seconds = summary["iteration_seconds"]
    phase = summary["phase_shares"]
    lines = [
        "# Compute-efficiency report",
        "",
        "## Steady-state throughput",
        "",
        (
            f"- Samples/s median: **{throughput['median']:,.0f}** "
            f"(p10 {throughput['p10']:,.0f}, p90 {throughput['p90']:,.0f})"
        ),
        f"- Seconds/iteration median: **{seconds['median']:.3f}** (p10 {seconds['p10']:.3f}, p90 {seconds['p90']:.3f})",
        f"- Timed iterations: {summary['steady_state_iterations']} (iterations 1–5 excluded)",
        "",
        "## Synchronized phase profile",
        "",
        f"- Rollout: {phase['rollout']:.1%}",
        f"- PPO update: {phase['update']:.1%}",
        f"- Host/bookkeeping gap: {phase['host']:.1%}",
        (
            "- Warm compilation cache reduced the first timed iteration by "
            f"{summary['compilation_cache_first_iteration_savings_seconds']:.1f} s"
        ),
        "",
        "## GPU telemetry",
        "",
    ]
    gpu = summary["gpu"]
    if gpu:
        if "gpu_util_pct" in gpu:
            lines.append(
                "- GPU utilization median/peak: "
                f"{gpu['gpu_util_pct']['median']:.1f}% / "
                f"{gpu['gpu_util_pct']['maximum']:.1f}%"
            )
        if "memory_used_mib" in gpu:
            lines.append(f"- Peak memory: {gpu['memory_used_mib']['maximum']:,.0f} MiB")
        if "peak_memory_fraction" in gpu:
            lines.append(f"- Peak memory fraction: {gpu['peak_memory_fraction']:.1%}")
        if "power_w" in gpu:
            lines.append(
                f"- Power median/peak: {gpu['power_w']['median']:.0f} W / {gpu['power_w']['maximum']:.0f} W"
            )
    else:
        lines.append("- GPU telemetry unavailable")
    lines.extend(["", "## Out-of-band events", ""])
    events = summary["event_timings_seconds"]
    if events:
        for name, value in sorted(events.items()):
            lines.append(f"- {name.replace('_', ' ').title()}: {value:.3f} s")
    else:
        lines.append("- No separately timed evaluation, pool, or checkpoint events")
    lines.extend(["", "## Device trace", "", f"- {summary['trace_status']}"])
    lines.extend(["", "## Recommendations", ""])
    for item in summary["recommendations"]:
        lines.append(
            f"- **{item['classification']} — {item['area']}:** {item['evidence']}"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    if summary["cprofile_excerpt"]:
        lines.extend(
            [
                "",
                "## Python profile excerpt",
                "",
                "```text",
                summary["cprofile_excerpt"],
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-metrics", type=Path, required=True)
    parser.add_argument("--phase-metrics", type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path, required=True)
    parser.add_argument("--cprofile-text", type=Path)
    parser.add_argument("--trace-status", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = summarize(args)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()

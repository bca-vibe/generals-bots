"""Fit a connected cross-seed temperature-strength model over the complete leagues."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit, logit


ROOT = Path(__file__).resolve().parent
INPUTS = (
    ROOT / "coarse_14k/round_robin.json",
    ROOT / "fine_025_050/round_robin.json",
    ROOT / "low_000_025/round_robin.json",
)


def temperature_label(value: float) -> str:
    return "greedy" if value == 0.0 else f"T={value:g}"


def load_records():
    records = []
    for path in INPUTS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload["complete"]:
            raise RuntimeError(f"Incomplete input: {path}")
        temperatures = {
            item["name"]: 0.0 if item["temperature"] is None else float(item["temperature"])
            for item in payload["participants"]
        }
        for match in payload["matches"]:
            score = float(np.clip(match["score"], 1e-4, 1.0 - 1e-4))
            score_se = match["paired_score_std"] / math.sqrt(payload["maps_per_matchup"])
            logit_variance = (score_se / (score * (1.0 - score))) ** 2
            records.append(
                {
                    "experiment": path.parent.name,
                    "temperature_a": temperatures[match["a"]],
                    "temperature_b": temperatures[match["b"]],
                    "score_a": score,
                    "logit_score_a": float(logit(score)),
                    "logit_variance": logit_variance,
                    "wins_a": match["wins"],
                    "losses_a": match["losses"],
                    "draws": match["draws"],
                    "games": match["games"],
                }
            )
    return records


def fit(records):
    temperatures = sorted(
        {record["temperature_a"] for record in records}
        | {record["temperature_b"] for record in records}
    )
    anchor = 0.0
    free = [temperature for temperature in temperatures if temperature != anchor]
    index = {temperature: position for position, temperature in enumerate(free)}
    design = []
    targets = []
    weights = []
    for record in records:
        row = np.zeros(len(free), dtype=np.float64)
        if record["temperature_a"] != anchor:
            row[index[record["temperature_a"]]] += 1.0
        if record["temperature_b"] != anchor:
            row[index[record["temperature_b"]]] -= 1.0
        design.append(row)
        targets.append(record["logit_score_a"])
        weights.append(1.0 / record["logit_variance"])
    design = np.asarray(design)
    targets = np.asarray(targets)
    weights = np.asarray(weights)
    information = design.T @ (weights[:, None] * design)
    coefficients = np.linalg.solve(information, design.T @ (weights * targets))
    residuals = targets - design @ coefficients
    chi_squared = float(np.sum(weights * residuals**2))
    degrees_of_freedom = len(targets) - len(coefficients)
    dispersion = max(1.0, chi_squared / degrees_of_freedom)
    covariance = np.linalg.inv(information) * dispersion

    rows = []
    for temperature in temperatures:
        if temperature == anchor:
            strength = 0.0
            standard_error = 0.0
        else:
            position = index[temperature]
            strength = float(coefficients[position])
            standard_error = math.sqrt(float(covariance[position, position]))
        rows.append(
            {
                "temperature": temperature,
                "label": temperature_label(temperature),
                "logit_strength_vs_greedy": strength,
                "logit_standard_error": standard_error,
                "modeled_score_vs_greedy": float(expit(strength)),
                "modeled_score_vs_greedy_ci95_low": float(expit(strength - 1.96 * standard_error)),
                "modeled_score_vs_greedy_ci95_high": float(expit(strength + 1.96 * standard_error)),
            }
        )
    rows.sort(key=lambda row: row["logit_strength_vs_greedy"], reverse=True)
    return rows, chi_squared, degrees_of_freedom, dispersion


def repeated_pairs(records):
    grouped = defaultdict(lambda: {"wins_low": 0.0, "losses_low": 0.0, "draws": 0.0, "games": 0.0, "experiments": []})
    for record in records:
        low, high = sorted((record["temperature_a"], record["temperature_b"]))
        row = grouped[(low, high)]
        if record["temperature_a"] == low:
            row["wins_low"] += record["wins_a"]
            row["losses_low"] += record["losses_a"]
        else:
            row["wins_low"] += record["losses_a"]
            row["losses_low"] += record["wins_a"]
        row["draws"] += record["draws"]
        row["games"] += record["games"]
        row["experiments"].append(record["experiment"])
    output = []
    for (low, high), row in grouped.items():
        if len(row["experiments"]) < 2:
            continue
        output.append(
            {
                "temperature_low": low,
                "temperature_high": high,
                **row,
                "score_low": (row["wins_low"] + 0.5 * row["draws"]) / row["games"],
            }
        )
    return sorted(output, key=lambda row: (row["temperature_low"], row["temperature_high"]))


def write_svg(rows):
    width = 940
    row_height = 38
    top = 80
    height = top + len(rows) * row_height + 55
    x0 = 290
    plot_width = 560
    low, high = 0.44, 0.54
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#222}.title{font-size:21px;font-weight:700}.sub{font-size:13px;fill:#555}.label{font-size:14px}.value{font-size:12px;font-weight:700}.axis{font-size:11px;fill:#666}</style>',
        '<text class="title" x="24" y="30">Combined cross-seed temperature model</text>',
        '<text class="sub" x="24" y="52">Modeled score versus greedy; bars show paired-map 95% intervals</text>',
    ]
    for tick in (0.44, 0.46, 0.48, 0.50, 0.52, 0.54):
        x = x0 + (tick - low) / (high - low) * plot_width
        parts.append(f'<line x1="{x}" x2="{x}" y1="{top - 14}" y2="{height - 38}" stroke="#{"999" if tick == 0.5 else "ddd"}"/>')
        parts.append(f'<text class="axis" x="{x}" y="{height - 18}" text-anchor="middle">{tick:.2f}</text>')
    for position, row in enumerate(rows):
        y = top + position * row_height
        score = row["modeled_score_vs_greedy"]
        ci_low = row["modeled_score_vs_greedy_ci95_low"]
        ci_high = row["modeled_score_vs_greedy_ci95_high"]
        score_x = x0 + (score - low) / (high - low) * plot_width
        low_x = x0 + (ci_low - low) / (high - low) * plot_width
        high_x = x0 + (ci_high - low) / (high - low) * plot_width
        parts.append(f'<text class="label" x="24" y="{y + 18}">{position + 1}. {row["label"]}</text>')
        parts.append(f'<line x1="{low_x}" x2="{high_x}" y1="{y + 13}" y2="{y + 13}" stroke="#334155" stroke-width="3"/>')
        parts.append(f'<circle cx="{score_x}" cy="{y + 13}" r="6" fill="#2563eb"/>')
        parts.append(f'<text class="value" x="{high_x + 8}" y="{y + 17}">{score:.4f}</text>')
    parts.append("</svg>")
    (ROOT / "combined_temperature_ranking.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    records = load_records()
    rows, chi_squared, degrees_of_freedom, dispersion = fit(records)
    repeated = repeated_pairs(records)
    payload = {
        "schema": "combined_temperature_strength_model_v1",
        "inputs": [str(path.relative_to(ROOT)) for path in INPUTS],
        "matchup_records": len(records),
        "games": int(sum(record["games"] for record in records)),
        "method": "inverse-variance weighted least squares on paired-map score logits, anchored at greedy",
        "chi_squared": chi_squared,
        "degrees_of_freedom": degrees_of_freedom,
        "dispersion": dispersion,
        "ranking": rows,
        "repeated_direct_pairs": repeated,
    }
    (ROOT / "combined_temperature_model.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (ROOT / "combined_temperature_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if repeated:
        with (ROOT / "repeated_direct_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["temperature_low", "temperature_high", "wins_low", "losses_low", "draws", "games", "score_low", "experiments"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in repeated:
                writer.writerow({**row, "experiments": ";".join(row["experiments"])})
    write_svg(rows)


if __name__ == "__main__":
    main()

"""Combine the broad and fine 26k EMA temperature sweeps."""

from __future__ import annotations

import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
INPUTS = (ROOT / "broad.json", ROOT / "fine.json")


def logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_records() -> tuple[list[dict], dict[str, dict]]:
    records = []
    payloads = {}
    for path in INPUTS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[path.stem] = payload
        temperatures = {
            item["name"]: 0.0 if item["temperature"] is None else float(item["temperature"])
            for item in payload["participants"]
        }
        for match in payload["matches"]:
            score = min(max(float(match["score"]), 1e-6), 1.0 - 1e-6)
            score_se = float(match["paired_score_std"]) / math.sqrt(payload["maps_per_matchup"])
            records.append(
                {
                    "experiment": path.stem,
                    "temperature_a": temperatures[match["a"]],
                    "temperature_b": temperatures[match["b"]],
                    "logit_score": math.log(score / (1.0 - score)),
                    "logit_se": score_se / (score * (1.0 - score)),
                    "match": match,
                }
            )
    return records, payloads


def fit(records: list[dict]) -> tuple[list[dict], dict]:
    temperatures = sorted(
        {temperature for record in records for temperature in (record["temperature_a"], record["temperature_b"])}
    )
    free = [temperature for temperature in temperatures if temperature != 0.0]
    design = []
    response = []
    weights = []
    for record in records:
        design.append(
            [
                (1.0 if record["temperature_a"] == temperature else 0.0)
                - (1.0 if record["temperature_b"] == temperature else 0.0)
                for temperature in free
            ]
        )
        response.append(record["logit_score"])
        weights.append(1.0 / record["logit_se"] ** 2)
    matrix = np.asarray(design, dtype=np.float64)
    response_array = np.asarray(response, dtype=np.float64)
    weight_matrix = np.diag(np.asarray(weights, dtype=np.float64))
    normal_inverse = np.linalg.inv(matrix.T @ weight_matrix @ matrix)
    strengths = normal_inverse @ matrix.T @ weight_matrix @ response_array
    residuals = response_array - matrix @ strengths
    chi_squared = float(residuals @ weight_matrix @ residuals)
    degrees_of_freedom = len(records) - len(free)
    dispersion = max(1.0, chi_squared / degrees_of_freedom)
    covariance = normal_inverse * dispersion

    rows = [
        {
            "temperature": 0.0,
            "label": "greedy",
            "logit_strength_vs_greedy": 0.0,
            "logit_standard_error": 0.0,
            "modeled_score_vs_greedy": 0.5,
            "modeled_score_vs_greedy_ci95_low": 0.5,
            "modeled_score_vs_greedy_ci95_high": 0.5,
        }
    ]
    for index, (temperature, strength) in enumerate(zip(free, strengths, strict=True)):
        standard_error = math.sqrt(covariance[index, index])
        rows.append(
            {
                "temperature": temperature,
                "label": f"T={temperature:g}",
                "logit_strength_vs_greedy": float(strength),
                "logit_standard_error": standard_error,
                "modeled_score_vs_greedy": logistic(float(strength)),
                "modeled_score_vs_greedy_ci95_low": logistic(float(strength) - 1.96 * standard_error),
                "modeled_score_vs_greedy_ci95_high": logistic(float(strength) + 1.96 * standard_error),
            }
        )
    rows.sort(key=lambda row: row["logit_strength_vs_greedy"], reverse=True)
    diagnostics = {
        "chi_squared": chi_squared,
        "degrees_of_freedom": degrees_of_freedom,
        "dispersion": dispersion,
    }
    return rows, diagnostics


def repeated_pairs(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        pair = tuple(sorted((record["temperature_a"], record["temperature_b"])))
        groups[pair].append(record)
    rows = []
    for (low, high), items in sorted(groups.items()):
        experiments = {item["experiment"] for item in items}
        if len(experiments) < 2:
            continue
        wins = losses = draws = games = 0.0
        for item in items:
            match = item["match"]
            if item["temperature_a"] == low:
                wins += match["wins"]
                losses += match["losses"]
            else:
                wins += match["losses"]
                losses += match["wins"]
            draws += match["draws"]
            games += match["games"]
        rows.append(
            {
                "temperature_low": low,
                "temperature_high": high,
                "wins_low": wins,
                "losses_low": losses,
                "draws": draws,
                "games": games,
                "score_low": (wins + 0.5 * draws) / games,
                "experiments": ";".join(sorted(experiments)),
            }
        )
    return rows


def flatten(payloads: dict[str, dict]) -> list[dict]:
    rows = []
    for experiment, payload in payloads.items():
        temperatures = {
            item["name"]: 0.0 if item["temperature"] is None else float(item["temperature"])
            for item in payload["participants"]
        }
        for match in payload["matches"]:
            low, high = match["score_ci95"]
            row = {
                "experiment": experiment,
                "temperature_a": temperatures[match["a"]],
                "temperature_b": temperatures[match["b"]],
                **{key: value for key, value in match.items() if key != "score_ci95"},
                "score_ci95_low": low,
                "score_ci95_high": high,
            }
            rows.append(row)
    return rows


def write_svg(rows: list[dict]) -> None:
    width, height = 900, 70 + 44 * len(rows)
    left, right = 170, 90
    chart_width = width - left - right
    minimum, maximum = 0.43, 0.53
    x = lambda score: left + (score - minimum) / (maximum - minimum) * chart_width
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#1f2937}.title{font-size:20px;font-weight:700}.label{font-size:14px}.value{font-size:13px;font-weight:600}.axis{stroke:#d1d5db;stroke-width:1}.ci{stroke:#111827;stroke-width:2}</style>',
        '<text x="24" y="30" class="title">26k EMA temperature model versus greedy</text>',
        f'<line x1="{x(.5):.1f}" y1="48" x2="{x(.5):.1f}" y2="{height-20}" class="axis"/>',
    ]
    for index, row in enumerate(rows):
        yy = 58 + 44 * index
        score = row["modeled_score_vs_greedy"]
        low = row["modeled_score_vs_greedy_ci95_low"]
        high = row["modeled_score_vs_greedy_ci95_high"]
        color = "#2563eb" if row["temperature"] in (0.05, 0.15) else "#64748b"
        parts.extend(
            [
                f'<text x="24" y="{yy+18}" class="label">{html.escape(row["label"])}</text>',
                f'<line x1="{x(low):.1f}" y1="{yy+11}" x2="{x(high):.1f}" y2="{yy+11}" class="ci"/>',
                f'<circle cx="{x(score):.1f}" cy="{yy+11}" r="7" fill="{color}"/>',
                f'<text x="{x(high)+10:.1f}" y="{yy+16}" class="value">{score:.2%}</text>',
            ]
        )
    parts.append("</svg>")
    (ROOT / "combined_ranking.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    records, payloads = load_records()
    rows, diagnostics = fit(records)
    repeats = repeated_pairs(records)
    payload = {
        "schema": "ema26k_temperature_model_v1",
        "inputs": [path.name for path in INPUTS],
        "matchup_records": len(records),
        "games": int(sum(record["match"]["games"] for record in records)),
        "method": "inverse-variance weighted least squares on paired-map score logits, anchored at greedy",
        **diagnostics,
        "ranking": rows,
        "repeated_direct_pairs": repeats,
    }
    (ROOT / "combined_model.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(ROOT / "combined_ranking.csv", rows)
    write_csv(ROOT / "repeated_direct_pairs.csv", repeats)
    all_rows = flatten(payloads)
    preferred = ["experiment", "temperature_a", "temperature_b", "a", "b", "wins", "losses", "draws", "games", "score", "score_ci95_low", "score_ci95_high", "paired_score_std", "evaluation_seconds"]
    behavior = sorted(key for key in all_rows[0] if key.startswith("behavior_"))
    write_csv(ROOT / "matchups.csv", all_rows, preferred + behavior)
    for experiment, stage in payloads.items():
        write_csv(ROOT / f"{experiment}_ranking.csv", stage["ranking"])
    write_svg(rows)
    print(json.dumps({"matchups": len(records), "games": payload["games"], "winner": rows[0]["label"]}, sort_keys=True))


if __name__ == "__main__":
    main()

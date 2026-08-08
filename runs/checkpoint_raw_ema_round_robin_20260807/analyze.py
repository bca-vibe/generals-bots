"""Generate compact tables and SVGs from the raw/EMA checkpoint league."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def perspective(match: dict, name: str) -> tuple[float, float, float, float]:
    if match["a"] == name:
        return match["score"], match["wins"], match["losses"], match["draws"]
    if match["b"] == name:
        return 1.0 - match["score"], match["losses"], match["wins"], match["draws"]
    raise ValueError(f"{name} is not in matchup")


def perspective_ci(match: dict, name: str) -> tuple[float, float]:
    if match["a"] == name:
        return tuple(match["score_ci95"])
    if match["b"] == name:
        return 1.0 - match["score_ci95"][1], 1.0 - match["score_ci95"][0]
    raise ValueError(f"{name} is not in matchup")


def find_match(matches: list[dict], first: str, second: str) -> dict:
    for match in matches:
        if {match["a"], match["b"]} == {first, second}:
            return match
    raise KeyError((first, second))


def flatten_match(match: dict, experiment: str) -> dict:
    row = {"experiment": experiment, **match}
    low, high = row.pop("score_ci95")
    row["score_ci95_low"] = low
    row["score_ci95_high"] = high
    return row


def ranking_svg(rows: list[dict]) -> None:
    width, height = 900, 72 + 42 * len(rows)
    left, right = 190, 70
    chart_width = width - left - right
    minimum, maximum = 0.40, 0.58
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#1f2937}.label{font-size:14px}.value{font-size:13px;font-weight:600}.title{font-size:20px;font-weight:700}.axis{stroke:#d1d5db;stroke-width:1}</style>',
        '<text x="24" y="30" class="title">14k–19k greedy raw/EMA league</text>',
    ]
    zero = left + (0.5 - minimum) / (maximum - minimum) * chart_width
    parts.append(f'<line x1="{zero:.1f}" y1="48" x2="{zero:.1f}" y2="{height-24}" class="axis"/>')
    for index, row in enumerate(rows):
        y = 58 + index * 42
        score = float(row["macro_score"])
        x = left + (score - minimum) / (maximum - minimum) * chart_width
        start, bar_width = min(zero, x), abs(x - zero)
        color = "#2563eb" if row["policy"] == "ema" else "#f59e0b"
        parts.extend(
            [
                f'<text x="24" y="{y+17}" class="label">{html.escape(row["name"])}</text>',
                f'<rect x="{start:.1f}" y="{y}" width="{max(bar_width,1):.1f}" height="22" rx="3" fill="{color}"/>',
                f'<text x="{x + (7 if score >= .5 else -7):.1f}" y="{y+16}" text-anchor="{"start" if score >= .5 else "end"}" class="value">{score:.3%}</text>',
            ]
        )
    parts.append('</svg>')
    (ROOT / "ranking.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def trend_svg(rows: list[dict]) -> None:
    width, height = 900, 520
    left, right, top, bottom = 80, 40, 60, 70
    chart_width, chart_height = width - left - right, height - top - bottom
    minimum, maximum = 0.42, 0.58
    iterations = sorted({int(row["iteration"]) for row in rows})
    x = lambda iteration: left + (iteration - min(iterations)) / (max(iterations) - min(iterations)) * chart_width
    y = lambda score: top + (maximum - score) / (maximum - minimum) * chart_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#1f2937}.title{font-size:20px;font-weight:700}.axis{stroke:#d1d5db;stroke-width:1}.tick{font-size:13px}.legend{font-size:14px;font-weight:600}</style>',
        '<text x="24" y="30" class="title">Checkpoint strength by policy tree</text>',
    ]
    for score in (0.42, 0.46, 0.50, 0.54, 0.58):
        yy = y(score)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" class="axis"/>')
        parts.append(f'<text x="{left-10}" y="{yy+5:.1f}" text-anchor="end" class="tick">{score:.0%}</text>')
    for iteration in iterations:
        xx = x(iteration)
        parts.append(f'<text x="{xx:.1f}" y="{height-34}" text-anchor="middle" class="tick">{iteration//1000}k</text>')
    for policy, color in (("raw", "#f59e0b"), ("ema", "#2563eb")):
        selected = sorted((row for row in rows if row["policy"] == policy), key=lambda row: row["iteration"])
        points = " ".join(f'{x(int(row["iteration"])):.1f},{y(float(row["macro_score"])):.1f}' for row in selected)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4"/>')
        for row in selected:
            xx, yy = x(int(row["iteration"])), y(float(row["macro_score"]))
            parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="6" fill="{color}"/>')
        parts.append(f'<rect x="{610 if policy == "raw" else 720}" y="20" width="16" height="16" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{632 if policy == "raw" else 742}" y="34" class="legend">{policy.upper()}</text>')
    parts.append('</svg>')
    (ROOT / "checkpoint_trend.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    league = json.loads((ROOT / "round_robin.json").read_text(encoding="utf-8"))
    followup = json.loads((ROOT / "followup_20k.json").read_text(encoding="utf-8"))
    matches = league["matches"]

    all_matchups = [flatten_match(match, "14k_19k_league") for match in matches]
    all_matchups.extend(flatten_match(match, "20k_followup") for match in followup["matches"])
    fields = ["experiment", "a", "b", "wins", "losses", "draws", "games", "score", "score_ci95_low", "score_ci95_high", "paired_score_std", "evaluation_seconds"]
    behavior_fields = sorted(key for key in all_matchups[0] if key.startswith("behavior_"))
    write_csv(ROOT / "matchups.csv", all_matchups, fields + behavior_fields)

    metadata = {item["name"]: item for item in league["participants"]}
    ranking = []
    for rank, row in enumerate(league["ranking"], 1):
        item = metadata[row["name"]]
        ranking.append(
            {
                "rank": rank,
                "name": row["name"],
                "iteration": item["iteration"],
                "policy": item["policy"],
                "macro_score": row["macro_score"],
                "wins": row["wins"],
                "losses": row["losses"],
                "draws": row["draws"],
                "games": row["games"],
            }
        )
    write_csv(ROOT / "ranking.csv", ranking)
    ranking_svg(ranking)

    trend = sorted(
        (
            {
                "iteration": row["iteration"],
                "policy": row["policy"],
                "name": row["name"],
                "macro_score": row["macro_score"],
            }
            for row in ranking
        ),
        key=lambda row: (row["iteration"], row["policy"]),
    )
    write_csv(ROOT / "checkpoint_trend.csv", trend)
    trend_svg(trend)

    same_checkpoint = []
    for iteration in range(14000, 20000, 1000):
        raw, ema = f"c{iteration}_raw", f"c{iteration}_ema"
        match = find_match(matches, raw, ema)
        score, wins, losses, draws = perspective(match, ema)
        same_checkpoint.append(
            {
                "iteration": iteration,
                "ema_score": score,
                "ema_wins": wins,
                "ema_losses": losses,
                "draws": draws,
                "games": match["games"],
                "ema_score_ci95_low": 1.0 - match["score_ci95"][1],
                "ema_score_ci95_high": 1.0 - match["score_ci95"][0],
            }
        )
    write_csv(ROOT / "same_checkpoint_raw_ema.csv", same_checkpoint)

    same_policy = []
    iterations = tuple(range(14000, 20000, 1000))
    for policy in ("raw", "ema"):
        for older_index, older in enumerate(iterations):
            for newer in iterations[older_index + 1 :]:
                older_name = f"c{older}_{policy}"
                newer_name = f"c{newer}_{policy}"
                match = find_match(matches, older_name, newer_name)
                score, wins, losses, draws = perspective(match, newer_name)
                low, high = perspective_ci(match, newer_name)
                same_policy.append(
                    {
                        "policy": policy,
                        "older_iteration": older,
                        "newer_iteration": newer,
                        "newer_score": score,
                        "newer_wins": wins,
                        "newer_losses": losses,
                        "draws": draws,
                        "games": match["games"],
                        "newer_score_ci95_low": low,
                        "newer_score_ci95_high": high,
                    }
                )
    write_csv(ROOT / "same_policy_progression.csv", same_policy)

    checkpoint_pairs = []
    for older_index, older in enumerate(iterations):
        for newer in iterations[older_index + 1 :]:
            wins = losses = draws = 0.0
            for older_policy in ("raw", "ema"):
                for newer_policy in ("raw", "ema"):
                    newer_name = f"c{newer}_{newer_policy}"
                    match = find_match(
                        matches, f"c{older}_{older_policy}", newer_name
                    )
                    _, match_wins, match_losses, match_draws = perspective(
                        match, newer_name
                    )
                    wins += match_wins
                    losses += match_losses
                    draws += match_draws
            games = wins + losses + draws
            checkpoint_pairs.append(
                {
                    "older_iteration": older,
                    "newer_iteration": newer,
                    "newer_score_across_raw_ema_cells": (wins + 0.5 * draws) / games,
                    "newer_wins": wins,
                    "newer_losses": losses,
                    "draws": draws,
                    "games": games,
                }
            )
    write_csv(ROOT / "checkpoint_pair_aggregate.csv", checkpoint_pairs)

    labels = league["matrices"]["labels"]
    matrix_rows = []
    for label, scores in zip(labels, league["matrices"]["score"], strict=True):
        matrix_rows.append({"row_policy": label, **dict(zip(labels, scores, strict=True))})
    write_csv(ROOT / "score_matrix.csv", matrix_rows)

    follow_rows = [flatten_match(match, "20k_followup") for match in followup["matches"]]
    write_csv(ROOT / "followup_20k.csv", follow_rows, fields + behavior_fields)


if __name__ == "__main__":
    main()

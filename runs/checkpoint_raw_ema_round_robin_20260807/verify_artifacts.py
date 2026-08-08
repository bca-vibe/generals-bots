"""Validate the raw/EMA checkpoint archive and refresh its SHA-256 manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def csv_rows(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_match(match: dict) -> None:
    assert match["games"] == 1024
    assert match["wins"] + match["losses"] + match["draws"] == 1024
    assert len(match["score_ci95"]) == 2
    assert match["score_ci95"][0] <= match["score"] <= match["score_ci95"][1]
    assert any(key.startswith("behavior_a_") for key in match)
    assert any(key.startswith("behavior_b_") for key in match)
    assert match["behavior_a_completed_games"] == 1024
    assert match["behavior_b_completed_games"] == 1024


def main() -> None:
    league = json.loads((ROOT / "round_robin.json").read_text(encoding="utf-8"))
    assert league["schema"] == "checkpoint_raw_ema_greedy_round_robin_v1"
    assert league["complete"] is True
    assert len(league["participants"]) == 12
    assert league["completed_matchups"] == league["total_matchups"] == 66
    assert len(league["matches"]) == 66
    assert league["games_per_matchup"] == 1024
    assert league["total_games"] == 67_584
    assert len(league["ranking"]) == 12
    assert league["ranking"][0]["name"] == "c19000_ema"
    for match in league["matches"]:
        validate_match(match)

    followup = json.loads((ROOT / "followup_20k.json").read_text(encoding="utf-8"))
    assert followup["schema"] == "checkpoint_20k_followup_v1"
    assert len(followup["participants"]) == 3
    assert len(followup["matches"]) == 2
    assert followup["games_per_matchup"] == 1024
    assert followup["total_games"] == 2_048
    for match in followup["matches"]:
        validate_match(match)
    assert followup["matches"][1]["a"] == "c20000_ema"
    assert followup["matches"][1]["b"] == "c19000_ema"
    assert followup["matches"][1]["score_ci95"][0] > 0.5

    assert len(csv_rows("matchups.csv")) == 68
    assert len(csv_rows("ranking.csv")) == 12
    assert len(csv_rows("checkpoint_trend.csv")) == 12
    assert len(csv_rows("same_checkpoint_raw_ema.csv")) == 6
    assert len(csv_rows("same_policy_progression.csv")) == 30
    assert len(csv_rows("checkpoint_pair_aggregate.csv")) == 15
    assert len(csv_rows("score_matrix.csv")) == 12
    assert len(csv_rows("followup_20k.csv")) == 2
    ET.parse(ROOT / "ranking.svg")
    ET.parse(ROOT / "checkpoint_trend.svg")
    log = (ROOT / "run.log").read_text(encoding="utf-8")
    assert "[66/66]" in log
    assert "FINAL RANKING" in log

    output = ROOT / "SHA256SUMS"
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != output
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    output.write_text(
        "\n".join(f"{digest(path)}  {path.relative_to(ROOT)}" for path in files)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "league_matchups": 66,
                "followup_matchups": 2,
                "total_matchups": 68,
                "total_games": 69_632,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

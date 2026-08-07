"""Validate the archived evaluation bundle and refresh its integrity hashes."""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMPLETE = (
    ROOT / "coarse_14k",
    ROOT / "fine_025_050",
    ROOT / "low_000_025",
)
INCOMPLETE = ROOT / "incomplete_checkpoint_screen"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_sums(directory: Path, *, recursive: bool) -> None:
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    output = directory / "SHA256SUMS"
    files = sorted(
        path
        for path in candidates
        if path.is_file() and path != output
    )
    lines = [f"{digest(path)}  {path.relative_to(directory)}" for path in files]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    total_games = 0
    complete_matches = 0
    for directory in COMPLETE:
        payload = json.loads((directory / "round_robin.json").read_text(encoding="utf-8"))
        assert payload["complete"] is True
        assert payload["completed_matchups"] == payload["total_matchups"] == len(payload["matches"])
        assert payload["games_per_matchup"] == 1024
        for match in payload["matches"]:
            assert match["games"] == 1024
            assert len(match["score_ci95"]) == 2
            assert any(key.startswith("behavior_a_") for key in match)
            assert any(key.startswith("behavior_b_") for key in match)
        with (directory / "matchups.csv").open(newline="", encoding="utf-8") as handle:
            assert sum(1 for _ in csv.DictReader(handle)) == len(payload["matches"])
        ET.parse(directory / "ranking.svg")
        ET.parse(directory / "score_heatmap.svg")
        total_games += len(payload["matches"]) * payload["games_per_matchup"]
        complete_matches += len(payload["matches"])

    partial = json.loads((INCOMPLETE / "round_robin.json").read_text(encoding="utf-8"))
    assert partial["complete"] is False
    assert partial["completed_matchups"] == len(partial["matches"]) == 33
    assert partial["games_per_matchup"] == 1024
    for match in partial["matches"]:
        assert match["games"] == 1024
        assert any(key.startswith("behavior_a_") for key in match)
        assert any(key.startswith("behavior_b_") for key in match)
    total_games += len(partial["matches"]) * partial["games_per_matchup"]

    model = json.loads((ROOT / "combined_temperature_model.json").read_text(encoding="utf-8"))
    assert model["matchup_records"] == complete_matches == 40
    assert model["games"] == 40_960
    assert total_games == 74_752
    ET.parse(ROOT / "combined_temperature_ranking.svg")

    for directory in (*COMPLETE, INCOMPLETE):
        write_sums(directory, recursive=False)
    write_sums(ROOT, recursive=True)
    print(
        json.dumps(
            {
                "complete_matchups": complete_matches,
                "complete_games": model["games"],
                "incomplete_screen_matchups": len(partial["matches"]),
                "total_archived_games": total_games,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

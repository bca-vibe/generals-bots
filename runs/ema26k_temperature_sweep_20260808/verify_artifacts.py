"""Verify completeness, provenance, derived tables, and file hashes."""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKPOINT_SHA256 = "fa98e35d79b0d3637dbc90408da69d1b9b3c2113d1210f392b53e6f4fb37c81e"
WEIGHTS_SHA256 = "9fb390390e9317d69c6448fd23317e7ad5e70aa37545823b7043405eaa150bac"
EXPECTED = {
    "broad": {"participants": 5, "matchups": 10, "map_seed": 202608081, "action_seed": 202608082},
    "fine": {"participants": 6, "matchups": 15, "map_seed": 202608083, "action_seed": 202608084},
}


def rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_payload(stage: str, expected: dict[str, int]) -> None:
    payload = json.loads((ROOT / f"{stage}.json").read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert payload["checkpoint"] == 26000
    assert payload["checkpoint_policy"] == "ema"
    assert payload["maps_per_matchup"] == 512
    assert payload["games_per_matchup"] == 1024
    assert payload["locked_maps_across_matchups"] is True
    assert payload["seat_swapped"] is True
    assert payload["map_seed"] == expected["map_seed"]
    assert payload["action_seed"] == expected["action_seed"]
    assert len(payload["participants"]) == expected["participants"]
    assert payload["completed_matchups"] == expected["matchups"]
    assert payload["total_matchups"] == expected["matchups"]
    assert len(payload["matches"]) == expected["matchups"]
    for participant in payload["participants"]:
        assert participant["checkpoint_sha256"] == CHECKPOINT_SHA256
        assert participant["weights_sha256"] == WEIGHTS_SHA256
    for match in payload["matches"]:
        assert match["games"] == 1024
        assert match["wins"] + match["losses"] + match["draws"] == 1024
        assert "behavior_a_actions" in match and "behavior_b_actions" in match


def verify_hashes() -> None:
    manifest = ROOT / "SHA256SUMS"
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected_digest, filename = line.split("  ", 1)
        entries.append(filename)
        assert digest(ROOT / filename) == expected_digest, filename
    expected_files = sorted(
        path.name
        for path in ROOT.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    assert sorted(entries) == expected_files


def main() -> None:
    for stage, expected in EXPECTED.items():
        verify_payload(stage, expected)

    metadata = json.loads((ROOT / "export_metadata.json").read_text(encoding="utf-8"))
    assert metadata["iteration"] == 26000 and metadata["policy"] == "ema"
    assert metadata["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert metadata["weights_sha256"] == WEIGHTS_SHA256

    model = json.loads((ROOT / "combined_model.json").read_text(encoding="utf-8"))
    assert model["matchup_records"] == 25 and model["games"] == 25600
    assert model["ranking"][0]["temperature"] == 0.05
    assert len(model["ranking"]) == 8
    assert len(rows("matchups.csv")) == 25
    assert len(rows("broad_ranking.csv")) == 5
    assert len(rows("fine_ranking.csv")) == 6
    assert len(rows("combined_ranking.csv")) == 8
    assert len(rows("repeated_direct_pairs.csv")) == 3
    ET.parse(ROOT / "combined_ranking.svg")
    verify_hashes()
    print("verified: 25 matchups, 25,600 games, provenance, tables, SVG, and SHA-256 hashes")


if __name__ == "__main__":
    main()

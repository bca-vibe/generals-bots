"""Summarize the proposed castle potential on an archived counterfactual atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

from generals.training.potential import (
    ARMY_NEED_TEMPERATURE,
    CASTLE_ASSET_TEMPERATURE,
    CASTLE_HORIZON_EVENTS,
    CASTLE_POTENTIAL_SCALE,
    ENEMY_SAFETY_RADIUS,
    GARRISON_CENTER,
    GARRISON_TEMPERATURE,
    LAND_NEED_TEMPERATURE,
    MINIMUM_CASTLE_NEED,
)


def _quantiles(values):
    return {
        name: float(value)
        for name, value in zip(
            ("p05", "p25", "median", "p75", "p95"),
            np.quantile(values, (0.05, 0.25, 0.5, 0.75, 0.95)),
        )
    }


def analyze(path: Path) -> dict:
    raw = np.load(path)
    causal = (
        raw["result__outcome"][:, :, 1] - raw["result__outcome"][:, :, 0]
    ).mean(axis=1)
    army_margin = raw["feature__army_margin"].astype(np.float64)
    land_margin = raw["feature__land_margin"].astype(np.float64)
    garrison = raw["feature__post_build_garrison"].astype(np.float64)
    enemy_distance = raw["feature__distance_to_enemy_land_true"].astype(np.float64)
    turn = raw["feature__turn"].astype(np.float64)

    raw_need = expit(
        -army_margin / ARMY_NEED_TEMPERATURE
        - land_margin / LAND_NEED_TEMPERATURE
    )
    need = MINIMUM_CASTLE_NEED + (1.0 - MINIMUM_CASTLE_NEED) * raw_need
    garrison_quality = expit(
        (garrison - GARRISON_CENTER) / GARRISON_TEMPERATURE
    )
    safety = np.clip(
        (enemy_distance - 1.0) / (ENEMY_SAFETY_RADIUS - 1.0), 0.0, 1.0
    )
    # Castles produce on even turns strictly before the 1,200-turn cap.
    future_growth_events = np.maximum(1199.0 // 2.0 - turn // 2.0, 0.0)
    horizon = np.clip(future_growth_events / CASTLE_HORIZON_EVENTS, 0.0, 1.0)
    new_asset = need * garrison_quality * safety * horizon
    # This is the isolated new-castle contribution around zero castle margin.
    approximate_phi_increase = CASTLE_POTENTIAL_SCALE * np.tanh(
        new_asset / CASTLE_ASSET_TEMPERATURE
    )
    positive = causal > 0
    negative = causal < 0
    return {
        "states": int(len(causal)),
        "equation": (
            "0.05*tanh(land_margin/20) + "
            "0.05*tanh(castle_asset_margin/0.10)"
        ),
        "new_castle_asset": {
            "all": _quantiles(new_asset),
            "causally_positive_mean": float(new_asset[positive].mean()),
            "causally_negative_mean": float(new_asset[negative].mean()),
            "spearman_with_causal_score_delta": float(
                spearmanr(new_asset, causal).statistic
            ),
        },
        "approximate_immediate_phi_increase": {
            "all": _quantiles(approximate_phi_increase),
            "causally_positive_mean": float(
                approximate_phi_increase[positive].mean()
            ),
            "causally_negative_mean": float(
                approximate_phi_increase[negative].mean()
            ),
        },
        "causal_groups": {
            "positive": int(positive.sum()),
            "negative": int(negative.sum()),
            "neutral": int((causal == 0).sum()),
        },
        "note": (
            "The approximation isolates the newly built castle. Exact Phi also "
            "includes pre-existing castles and both players' land/castle margins."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.atlas)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

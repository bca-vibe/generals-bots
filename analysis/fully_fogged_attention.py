"""Measure attention when the spatial input contains fog and position only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from adjacent_attention import (
    capture_patterns,
    load_checkpoint,
    summarize_patterns,
)

from generals.training.config import TrainingConfig
from generals.training.observation import LEGACY_OBSERVATION_SCHEMA


def fully_fogged_inputs(config: TrainingConfig, samples: int):
    """Return content-free inputs with only fog and cell coordinates present.

    Public histories and every state/memory channel are zeroed. Keeping the
    coordinate channels matches the model's normal positional input, while the
    all-one fog channel represents a completely unobserved 21x21 board.
    """
    board_size = config.pad_to
    augmented = np.zeros(
        (samples, config.input_channels, board_size, board_size), dtype=np.float32
    )
    offset = 0 if config.observation_schema == LEGACY_OBSERVATION_SCHEMA else -1
    augmented[:, 12 + offset] = 1.0
    x_coord = np.broadcast_to(
        np.arange(board_size, dtype=np.float32)[None] / (board_size - 1),
        (board_size, board_size),
    )
    y_coord = np.broadcast_to(
        np.arange(board_size, dtype=np.float32)[:, None] / (board_size - 1),
        (board_size, board_size),
    )
    augmented[:, 22 + offset] = x_coord
    augmented[:, 23 + offset] = y_coord
    histories = np.zeros(
        (samples, 2, config.temporal_window), dtype=np.float32
    )
    groups = np.arange(samples, dtype=np.int32)
    return augmented, histories, groups


def center_patch_payload(patterns, samples, equivalence_error):
    """Average each head's center-query attention into visualization data."""
    center_spatial_index = 24
    center_token_index = 3 + center_spatial_index
    center = patterns[..., center_token_index, :].mean(axis=1)
    heads = []
    for layer in range(center.shape[0]):
        for head in range(center.shape[1]):
            heads.append(
                {
                    "layer": layer,
                    "head": head,
                    "special": np.round(center[layer, head, :3], 5).tolist(),
                    "spatial": np.round(center[layer, head, 3:], 5).tolist(),
                    "spatialMass": round(float(center[layer, head, 3:].sum()), 5),
                }
            )
    return {
        "samples": samples,
        "queryToken": center_token_index,
        "equivalenceError": equivalence_error,
        "condition": "fog-and-position-only",
        "heads": heads,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="generals/training/configs/smoke_8xh100.toml")
    parser.add_argument(
        "--checkpoint", default="checkpoints/smoke_8xh100/checkpoint_001260.eqx"
    )
    parser.add_argument("--output-dir", default="analysis/fully-fogged-attention")
    parser.add_argument(
        "--reference-metrics",
        default="runs/smoke_8xh100/attention_analysis/head_metrics.csv",
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    config = TrainingConfig.from_toml(args.config)
    network = load_checkpoint(config, Path(args.checkpoint))
    augmented, histories, groups = fully_fogged_inputs(config, args.samples)
    patterns, equivalence_error = capture_patterns(
        network, augmented, histories, args.layers
    )
    rows = summarize_patterns(
        patterns, groups, "iteration_1260_fully_fogged", args.seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "head_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(rows, key=lambda row: row["adjacent4_enrichment"], reverse=True)
    summary = {
        "checkpoint": args.checkpoint,
        "uses_ema_parameters": True,
        "condition": {
            "fog_channel": "one at every cell",
            "coordinate_channels": "retained",
            "all_other_spatial_channels": "zero",
            "army_and_land_histories": "zero",
            "purpose": "Remove board content while retaining positional information.",
        },
        "samples": args.samples,
        "transformer_lens_equivalence_max_abs_error": equivalence_error,
        "null_definition": (
            "Each query's non-self spatial attention mass redistributed uniformly "
            "over the other 48 spatial keys, preserving border degree."
        ),
        "top_heads": ranked[:10],
    }

    reference_path = Path(args.reference_metrics)
    if reference_path.exists():
        with reference_path.open(encoding="utf-8") as handle:
            reference_rows = [
                row
                for row in csv.DictReader(handle)
                if row["checkpoint"] == "iteration_1260"
            ]
        reference = {
            (int(row["layer"]), int(row["head"])): row for row in reference_rows
        }
        comparison = []
        for row in rows:
            key = (row["layer"], row["head"])
            real_enrichment = float(reference[key]["adjacent4_enrichment"])
            fogged_enrichment = row["adjacent4_enrichment"]
            comparison.append(
                {
                    "layer": row["layer"],
                    "head": row["head"],
                    "held_out_adjacent4_enrichment": real_enrichment,
                    "fully_fogged_adjacent4_enrichment": fogged_enrichment,
                    "difference": fogged_enrichment - real_enrichment,
                }
            )
        with (output_dir / "head_comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
        held_out = np.array(
            [row["held_out_adjacent4_enrichment"] for row in comparison]
        )
        fully_fogged = np.array(
            [row["fully_fogged_adjacent4_enrichment"] for row in comparison]
        )
        summary["comparison_to_held_out_observations"] = {
            "pearson_correlation_across_24_heads": float(
                np.corrcoef(held_out, fully_fogged)[0, 1]
            ),
            "heads_above_1_5x_in_both": [
                {"layer": row["layer"], "head": row["head"]}
                for row in comparison
                if row["held_out_adjacent4_enrichment"] > 1.5
                and row["fully_fogged_adjacent4_enrichment"] > 1.5
            ],
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "center_patch_payload.json").write_text(
        json.dumps(center_patch_payload(patterns, args.samples, equivalence_error)),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

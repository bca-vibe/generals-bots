"""Compare raw castle critics on matched-policy and common-action atlases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


def _source_groups(raw):
    return raw["metadata__source_batch"].astype(np.int64) * 512 + raw["metadata__source_slot"].astype(np.int64) // 12


def _cluster_ci(values, groups, seed, draws=10_000):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    mask = np.isfinite(values)
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=np.where(mask, values, 0.0))
    counts = np.bincount(inverse, weights=mask.astype(np.float64))
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, draws, 500):
        size = min(500, draws - start)
        sample = rng.integers(0, len(unique), (size, len(unique)))
        estimates.extend(sums[sample].sum(1) / counts[sample].sum(1))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _state_means(values, valid):
    output = np.full(values.shape[0], np.nan)
    for index in range(values.shape[0]):
        if valid[index].any():
            output[index] = values[index, valid[index]].mean()
    return output


def _half_means(values, valid, start, stop):
    return _state_means(values[:, start:stop], valid[:, start:stop])


def _correlation(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    return {
        "states": int(mask.sum()),
        "spearman": float(spearmanr(x[mask], y[mask]).statistic),
        "pearson": float(pearsonr(x[mask], y[mask]).statistic),
    }


def _matched_summary(path: Path, label: str, seed: int):
    raw = np.load(path)
    outcome = raw["result__outcome"].astype(np.float64)
    post = raw["result__post_actor_value"].astype(np.float64)
    pre = raw["feature__actor_value"].astype(np.float64)
    valid = ~raw["result__intervention_done"].any(axis=2)
    causal_pair = outcome[:, :, 1] - outcome[:, :, 0]
    value_pair = post[:, :, 1] - post[:, :, 0]
    build_pre_pair = post[:, :, 1] - pre[:, None]
    causal = causal_pair.mean(axis=1)
    value = _state_means(value_pair, valid)
    build_pre = _state_means(build_pre_pair, valid)
    groups = _source_groups(raw)
    half = outcome.shape[1] // 2
    causal_first = causal_pair[:, :half].mean(axis=1)
    causal_second = causal_pair[:, half:].mean(axis=1)
    value_first = _half_means(value_pair, valid, 0, half)
    value_second = _half_means(value_pair, valid, half, outcome.shape[1])
    heldout_value = np.concatenate([value_second[causal_first > 0], value_first[causal_second > 0]])
    heldout_causal = np.concatenate([causal_second[causal_first > 0], causal_first[causal_second > 0]])
    heldout_groups = np.concatenate([groups[causal_first > 0], groups[causal_second > 0]])
    causally_positive = causal > 0
    both_good = (causal_first > 0) & (causal_second > 0)
    finite = np.isfinite(value)
    return {
        "label": label,
        "states": int(len(value)),
        "causal": causal,
        "value": value,
        "build_pre": build_pre,
        "groups": groups,
        "both_good": both_good,
        "heldout_value": heldout_value,
        "heldout_causal": heldout_causal,
        "heldout_groups": heldout_groups,
        "total_build_probability": raw["feature__total_build_probability"],
        "best_build_logit_margin": raw["feature__best_build_logit_margin"],
        "summary": {
            "states_with_value": int(finite.sum()),
            "causal_score_delta": float(causal.mean()),
            "successor_value_delta": float(np.nanmean(value)),
            "successor_value_delta_ci95": _cluster_ci(value, groups, seed),
            "fraction_value_positive": float(np.mean(value[finite] > 0)),
            "build_successor_minus_pre": float(np.nanmean(build_pre)),
            "causally_positive_states": int(causally_positive.sum()),
            "causally_positive_causal_delta": float(causal[causally_positive].mean()),
            "causally_positive_value_delta": float(np.nanmean(value[causally_positive])),
            "causally_positive_value_ci95": _cluster_ci(value[causally_positive], groups[causally_positive], seed + 2),
            "stable_good_states": int(both_good.sum()),
            "stable_good_causal_delta": float(causal[both_good].mean()),
            "stable_good_value_delta": float(np.nanmean(value[both_good])),
            "stable_good_value_ci95": _cluster_ci(value[both_good], groups[both_good], seed + 3),
            "heldout_good_selections": int(np.isfinite(heldout_value).sum()),
            "heldout_good_causal_delta": float(np.nanmean(heldout_causal)),
            "heldout_good_value_delta": float(np.nanmean(heldout_value)),
            "heldout_good_value_ci95": _cluster_ci(heldout_value, heldout_groups, seed + 1),
            "fraction_heldout_value_positive": float(np.nanmean(heldout_value > 0)),
            "value_vs_causal": _correlation(value, causal),
            "median_total_build_probability": float(np.median(raw["feature__total_build_probability"])),
            "median_best_build_rank": float(np.median(raw["feature__best_build_rank"])),
            "median_build_logit_margin": float(np.median(raw["feature__best_build_logit_margin"])),
        },
    }


def _common_summary(path: Path, seed: int):
    raw = np.load(path)
    atlas = json.loads((path.parent / "atlas.json").read_text(encoding="utf-8"))
    labels = [item["name"] for item in atlas["common_action_critics"]]
    outcome = raw["result__outcome"].astype(np.float64)
    causal_pair = outcome[:, :, 1] - outcome[:, :, 0]
    causal = causal_pair.mean(axis=1)
    valid = ~raw["result__intervention_done"].any(axis=2)
    post = raw["result__common_critic_post_actor_values"].astype(np.float64)
    pre = raw["feature__common_critic_actor_values"].astype(np.float64)
    groups = _source_groups(raw)
    half = outcome.shape[1] // 2
    causal_first = causal_pair[:, :half].mean(axis=1)
    causal_second = causal_pair[:, half:].mean(axis=1)
    causally_positive = causal > 0
    both_good = (causal_first > 0) & (causal_second > 0)
    records = []
    state_values = {}
    for critic_index, label in enumerate(labels):
        value_pair = post[:, :, 1, critic_index] - post[:, :, 0, critic_index]
        build_pre_pair = post[:, :, 1, critic_index] - pre[:, critic_index, None]
        value = _state_means(value_pair, valid)
        build_pre = _state_means(build_pre_pair, valid)
        first = _half_means(value_pair, valid, 0, half)
        second = _half_means(value_pair, valid, half, outcome.shape[1])
        heldout_value = np.concatenate([second[causal_first > 0], first[causal_second > 0]])
        heldout_groups = np.concatenate([groups[causal_first > 0], groups[causal_second > 0]])
        state_values[label] = value
        records.append(
            {
                "label": label,
                "value": value,
                "build_pre": build_pre,
                "heldout_value": heldout_value,
                "summary": {
                    "successor_value_delta": float(np.nanmean(value)),
                    "successor_value_delta_ci95": _cluster_ci(value, groups, seed + critic_index),
                    "fraction_value_positive": float(np.mean(value[np.isfinite(value)] > 0)),
                    "build_successor_minus_pre": float(np.nanmean(build_pre)),
                    "causally_positive_states": int(causally_positive.sum()),
                    "causally_positive_causal_delta": float(causal[causally_positive].mean()),
                    "causally_positive_value_delta": float(np.nanmean(value[causally_positive])),
                    "causally_positive_value_ci95": _cluster_ci(
                        value[causally_positive],
                        groups[causally_positive],
                        seed + 200 + critic_index,
                    ),
                    "stable_good_states": int(both_good.sum()),
                    "stable_good_causal_delta": float(causal[both_good].mean()),
                    "stable_good_value_delta": float(np.nanmean(value[both_good])),
                    "stable_good_value_ci95": _cluster_ci(
                        value[both_good],
                        groups[both_good],
                        seed + 300 + critic_index,
                    ),
                    "heldout_good_value_delta": float(np.nanmean(heldout_value)),
                    "heldout_good_value_ci95": _cluster_ci(
                        heldout_value,
                        heldout_groups,
                        seed + 100 + critic_index,
                    ),
                    "fraction_heldout_value_positive": float(np.nanmean(heldout_value > 0)),
                    "value_vs_causal": _correlation(value, causal),
                },
            }
        )
    control = next(label for label in labels if "control" in label)
    treatment = next(label for label in labels if "treatment" in label)
    paired_shift = state_values[treatment] - state_values[control]
    return {
        "labels": labels,
        "causal": causal,
        "groups": groups,
        "both_good": both_good,
        "records": records,
        "paired_treatment_minus_control": paired_shift,
        "paired_treatment_minus_control_summary": {
            "mean": float(np.nanmean(paired_shift)),
            "ci95": _cluster_ci(paired_shift, groups, seed + 999),
            "fraction_positive": float(np.nanmean(paired_shift > 0)),
            "stable_good_mean": float(np.nanmean(paired_shift[both_good])),
            "stable_good_ci95": _cluster_ci(paired_shift[both_good], groups[both_good], seed + 1000),
        },
    }


def _errorbar(ax, x, mean, interval, **kwargs):
    return ax.errorbar(
        x,
        mean,
        yerr=[[mean - interval[0]], [interval[1] - mean]],
        capsize=4,
        **kwargs,
    )


def _charts(matched, common, output_dir: Path):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    all_handle = None
    heldout_handle = None
    for index, record in enumerate(common["records"]):
        summary = record["summary"]
        all_handle = _errorbar(
            ax,
            index - 0.12,
            summary["successor_value_delta"],
            summary["successor_value_delta_ci95"],
            fmt="o",
            color="#4C78A8",
        )
        heldout_handle = _errorbar(
            ax,
            index + 0.12,
            summary["heldout_good_value_delta"],
            summary["heldout_good_value_ci95"],
            fmt="o",
            color="#F58518",
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(common["records"])), common["labels"], rotation=20)
    ax.set_ylabel("V(build successor) − V(control successor)")
    ax.set_title("Common-action critic response by raw checkpoint")
    ax.legend(
        [all_handle, heldout_handle],
        ["All legal opportunities", "Held-out good builds"],
    )
    fig.tight_layout()
    fig.savefig(output_dir / "01_critic_response_by_checkpoint.png", dpi=180)
    plt.close(fig)

    shifts = common["paired_treatment_minus_control"]
    shifts = np.sort(shifts[np.isfinite(shifts)])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(shifts, np.arange(1, len(shifts) + 1) / len(shifts))
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Treatment ΔV − control ΔV")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_title("Paired φ-boost critic shift on identical successors")
    fig.tight_layout()
    fig.savefig(output_dir / "02_treatment_control_paired_shift.png", dpi=180)
    plt.close(fig)

    selected = [
        record
        for record in common["records"]
        if any(token in record["label"] for token in ("3000", "control", "treatment"))
    ]
    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4), sharex=True, sharey=True)
    if len(selected) == 1:
        axes = [axes]
    for ax, record in zip(axes, selected):
        ax.hexbin(common["causal"], record["value"], gridsize=35, mincnt=1, cmap="viridis")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(record["label"])
        ax.set_xlabel("Actual paired score effect")
    axes[0].set_ylabel("Critic successor-value difference")
    fig.suptitle("Causal castle effect versus critic prediction")
    fig.tight_layout()
    fig.savefig(output_dir / "03_causal_vs_critic.png", dpi=180)
    plt.close(fig)

    groups = ["All", "Stable good", "Held-out good"]
    x = np.arange(len(groups))
    width = 0.8 / len(matched)
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, record in enumerate(matched):
        values = [
            record["summary"]["successor_value_delta"],
            record["summary"]["stable_good_value_delta"],
            record["summary"]["heldout_good_value_delta"],
        ]
        ax.bar(x + (index - (len(matched) - 1) / 2) * width, values, width, label=record["label"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, groups)
    ax.set_ylabel("Mean successor-value difference")
    ax.set_title("Matched-policy critic response by causal group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04_matched_policy_groups.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(matched))
    ax.bar(
        x - 0.18,
        [record["summary"]["build_successor_minus_pre"] for record in matched],
        0.36,
        label="Build successor − pre",
    )
    ax.bar(
        x + 0.18,
        [record["summary"]["successor_value_delta"] for record in matched],
        0.36,
        label="Build successor − control successor",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, [record["label"] for record in matched])
    ax.set_ylabel("Value difference")
    ax.set_title("Value decomposition around the forced build")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "05_value_decomposition.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(matched), figsize=(5 * len(matched), 4), sharey=True)
    if len(matched) == 1:
        axes = [axes]
    for ax, record in zip(axes, matched):
        order = np.argsort(record["value"])
        bins = np.array_split(order, 10)
        ax.plot(
            range(1, 11),
            [np.nanmean(record["causal"][selection]) for selection in bins],
            marker="o",
            label="Causal effect",
        )
        ax.set_title(record["label"])
        ax.set_xlabel("Critic ΔV decile")
    axes[0].set_ylabel("Actual paired score effect")
    fig.suptitle("Policy–critic alignment: causal effect by critic ranking")
    fig.tight_layout()
    fig.savefig(output_dir / "06_policy_critic_alignment.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw3000", type=Path, required=True)
    parser.add_argument("--control4000", type=Path, required=True)
    parser.add_argument("--treatment4000", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched_internal = [
        _matched_summary(args.raw3000, "raw_3000", args.seed),
        _matched_summary(args.control4000, "control_4000", args.seed + 10),
        _matched_summary(args.treatment4000, "treatment_4000", args.seed + 20),
    ]
    common = _common_summary(args.raw3000, args.seed + 30)
    _charts(matched_internal, common, args.output_dir)
    report = {
        "method": {
            "matched_policy": "each target chooses interventions and continuation",
            "common_action": "raw 3000 actions, opponent actions, successors, and causal labels",
            "good_definition": "select on one 8-rollout half and evaluate on the other, then swap",
            "value_scale": "network expectation on [-1, 1]",
            "causal_scale": "game score on [0, 1]",
        },
        "matched_policy": {record["label"]: record["summary"] for record in matched_internal},
        "common_action": {record["label"]: record["summary"] for record in common["records"]},
        "paired_treatment_minus_control": common["paired_treatment_minus_control_summary"],
    }
    (args.output_dir / "critic_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

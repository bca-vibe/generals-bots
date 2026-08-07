"""Analyze successor-state value estimates in a castle counterfactual atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _cluster_mean_ci(values, mask, groups, seed, draws=10_000):
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=np.where(mask, values, 0.0))
    counts = np.bincount(inverse, weights=mask.astype(np.float64))
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, draws, 500):
        count = min(500, draws - start)
        sample = rng.integers(0, len(unique), size=(count, len(unique)))
        denominator = counts[sample].sum(axis=1)
        estimates.extend(sums[sample].sum(axis=1) / denominator)
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def _cluster_difference_ci(values, first, second, groups, seed, draws=10_000):
    values = np.asarray(values, dtype=np.float64)
    first = np.asarray(first, dtype=bool) & np.isfinite(values)
    second = np.asarray(second, dtype=bool) & np.isfinite(values)
    unique, inverse = np.unique(groups, return_inverse=True)
    first_sums = np.bincount(inverse, weights=np.where(first, values, 0.0))
    first_counts = np.bincount(inverse, weights=first.astype(np.float64))
    second_sums = np.bincount(inverse, weights=np.where(second, values, 0.0))
    second_counts = np.bincount(inverse, weights=second.astype(np.float64))
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, draws, 500):
        count = min(500, draws - start)
        sample = rng.integers(0, len(unique), size=(count, len(unique)))
        first_mean = first_sums[sample].sum(axis=1) / first_counts[sample].sum(
            axis=1
        )
        second_mean = second_sums[sample].sum(axis=1) / second_counts[
            sample
        ].sum(axis=1)
        estimates.extend(first_mean - second_mean)
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def _correlations(first, second):
    mask = np.isfinite(first) & np.isfinite(second)
    return {
        "states": int(mask.sum()),
        "spearman": float(spearmanr(first[mask], second[mask]).statistic),
        "pearson": float(pearsonr(first[mask], second[mask]).statistic),
    }


def _summary(name, mask, causal_delta, value_delta, build_pre, groups, seed):
    mask = np.asarray(mask, dtype=bool) & np.isfinite(value_delta)
    return {
        "name": name,
        "states": int(mask.sum()),
        "causal_score_delta": float(causal_delta[mask].mean()),
        "successor_value_delta_build_minus_control": float(
            value_delta[mask].mean()
        ),
        "successor_value_delta_cluster_bootstrap_95": _cluster_mean_ci(
            value_delta, mask, groups, seed
        ),
        "successor_value_delta_median": float(np.median(value_delta[mask])),
        "fraction_successor_value_delta_positive": float(
            np.mean(value_delta[mask] > 0)
        ),
        "build_successor_minus_pre_value": float(build_pre[mask].mean()),
        "fraction_build_successor_above_pre_value": float(
            np.mean(build_pre[mask] > 0)
        ),
    }


def analyze(atlas_path: Path, selfplay_path: Path, seed: int):
    raw = np.load(atlas_path)
    outcome = raw["result__outcome"].astype(np.float64)
    post_value = raw["result__post_actor_value"].astype(np.float64)
    pre_value = raw["feature__actor_value"].astype(np.float64)
    intervention_done = raw["result__intervention_done"].any(axis=2)
    valid_pair = ~intervention_done

    control = outcome[:, :, 0]
    build = outcome[:, :, 1]
    paired_outcome = build - control
    causal_delta = paired_outcome.mean(axis=1)
    pair_value_delta = post_value[:, :, 1] - post_value[:, :, 0]
    pair_build_pre = post_value[:, :, 1] - pre_value[:, None]
    pair_control_pre = post_value[:, :, 0] - pre_value[:, None]

    def valid_state_mean(values):
        result = np.full(len(values), np.nan)
        for index in range(len(values)):
            if np.any(valid_pair[index]):
                result[index] = values[index, valid_pair[index]].mean()
        return result

    value_delta = valid_state_mean(pair_value_delta)
    build_pre = valid_state_mean(pair_build_pre)
    control_pre = valid_state_mean(pair_control_pre)
    finite = np.isfinite(value_delta)

    source_batch = raw["metadata__source_batch"].astype(np.int64)
    source_slot = raw["metadata__source_slot"].astype(np.int64)
    source_game = source_batch * 512 + source_slot // 12

    half = outcome.shape[1] // 2
    causal_first = paired_outcome[:, :half].mean(axis=1)
    causal_second = paired_outcome[:, half:].mean(axis=1)

    def half_value_mean(values, start, stop):
        result = np.full(len(values), np.nan)
        for index in range(len(values)):
            mask = valid_pair[index, start:stop]
            if np.any(mask):
                result[index] = values[index, start:stop][mask].mean()
        return result

    value_first = half_value_mean(pair_value_delta, 0, half)
    value_second = half_value_mean(pair_value_delta, half, outcome.shape[1])
    select_first = causal_first > 0
    select_second = causal_second > 0
    heldout_value = np.concatenate(
        [value_second[select_first], value_first[select_second]]
    )
    heldout_causal = np.concatenate(
        [causal_second[select_first], causal_first[select_second]]
    )
    heldout_groups = np.concatenate(
        [source_game[select_first], source_game[select_second]]
    )
    heldout_finite = np.isfinite(heldout_value)

    good = causal_delta > 0
    bad = causal_delta < 0
    both_good = (causal_first > 0) & (causal_second > 0)
    both_bad = (causal_first < 0) & (causal_second < 0)

    order = np.flatnonzero(finite)[np.argsort(value_delta[finite])]
    decile = len(order) // 10
    bottom = order[:decile]
    top = order[-decile:]

    selfplay = json.loads(selfplay_path.read_text(encoding="utf-8"))
    return {
        "checkpoint_iteration": int(selfplay["checkpoint_iteration"]),
        "checkpoint_sha256": selfplay["checkpoint_sha256"],
        "value_scale": "network expectation on [-1, 1]",
        "data_integrity": {
            "states": int(len(outcome)),
            "paired_repetitions_per_state": int(outcome.shape[1]),
            "branch_rollouts": int(outcome.size),
            "intervention_pairs_terminal_immediately": int(
                intervention_done.sum()
            ),
            "states_excluded_from_successor_value_analysis": int(
                (~finite).sum()
            ),
            "source_games_represented": int(np.unique(source_game).size),
        },
        "overall": {
            "causal_score_delta": float(causal_delta.mean()),
            "pre_actor_value_mean": float(pre_value[finite].mean()),
            "control_successor_minus_pre_value": float(control_pre[finite].mean()),
            "build_successor_minus_pre_value": float(build_pre[finite].mean()),
            "successor_value_delta_build_minus_control": float(
                value_delta[finite].mean()
            ),
            "successor_value_delta_cluster_bootstrap_95": _cluster_mean_ci(
                value_delta, finite, source_game, seed
            ),
            "fraction_successor_value_delta_positive": float(
                np.mean(value_delta[finite] > 0)
            ),
            "fraction_build_successor_above_pre_value": float(
                np.mean(build_pre[finite] > 0)
            ),
        },
        "causal_groups": {
            "positive_full_sample": _summary(
                "positive full-sample causal estimate",
                good,
                causal_delta,
                value_delta,
                build_pre,
                source_game,
                seed + 1,
            ),
            "negative_full_sample": _summary(
                "negative full-sample causal estimate",
                bad,
                causal_delta,
                value_delta,
                build_pre,
                source_game,
                seed + 2,
            ),
            "positive_both_halves": _summary(
                "positive causal estimate in both rollout halves",
                both_good,
                causal_delta,
                value_delta,
                build_pre,
                source_game,
                seed + 3,
            ),
            "negative_both_halves": _summary(
                "negative causal estimate in both rollout halves",
                both_bad,
                causal_delta,
                value_delta,
                build_pre,
                source_game,
                seed + 4,
            ),
            "good_minus_bad_successor_value_delta": float(
                value_delta[good & finite].mean() - value_delta[bad & finite].mean()
            ),
            "good_minus_bad_cluster_bootstrap_95": _cluster_difference_ci(
                value_delta, good, bad, source_game, seed + 5
            ),
        },
        "heldout_good_definition": {
            "selection": (
                "classify a state as good on one 8-rollout half and evaluate "
                "value and causal effect on the other half, then swap"
            ),
            "state_half_selections": int(heldout_finite.sum()),
            "heldout_causal_score_delta": float(
                heldout_causal[heldout_finite].mean()
            ),
            "heldout_successor_value_delta": float(
                heldout_value[heldout_finite].mean()
            ),
            "heldout_successor_value_delta_cluster_bootstrap_95": (
                _cluster_mean_ci(
                    heldout_value,
                    heldout_finite,
                    heldout_groups,
                    seed + 6,
                )
            ),
            "fraction_heldout_successor_value_delta_positive": float(
                np.mean(heldout_value[heldout_finite] > 0)
            ),
        },
        "ranking_signal": {
            "full_state_value_delta_vs_causal_delta": _correlations(
                value_delta, causal_delta
            ),
            "first_half_value_vs_second_half_causal": _correlations(
                value_first, causal_second
            ),
            "second_half_value_vs_first_half_causal": _correlations(
                value_second, causal_first
            ),
            "least_bad_value_decile": {
                "states": int(len(top)),
                "value_delta": float(value_delta[top].mean()),
                "causal_score_delta": float(causal_delta[top].mean()),
                "fraction_causally_positive": float(np.mean(causal_delta[top] > 0)),
            },
            "worst_value_decile": {
                "states": int(len(bottom)),
                "value_delta": float(value_delta[bottom].mean()),
                "causal_score_delta": float(causal_delta[bottom].mean()),
                "fraction_causally_positive": float(
                    np.mean(causal_delta[bottom] > 0)
                ),
            },
        },
        "selfplay_game_length": selfplay["policies"].get(
            "raw", selfplay["policies"].get("ema")
        )["game_length"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--selfplay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    report = analyze(args.atlas, args.selfplay, args.seed)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

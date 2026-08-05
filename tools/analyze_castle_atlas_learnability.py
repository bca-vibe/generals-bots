"""Cross-fit simple selectors on a castle counterfactual atlas.

The paired rollout repetitions are split in half. Models learn causal score
deltas from one half and are evaluated on the other half for held-out source
games. Swapping the two halves gives a symmetric estimate. This keeps both
state identity and rollout noise out of the evaluation labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OBSERVABLE_FEATURES = (
    "turn",
    "site_cost",
    "site_army",
    "post_build_garrison",
    "own_army",
    "opponent_army",
    "army_margin",
    "own_land",
    "opponent_land",
    "land_margin",
    "own_castles",
    "distance_to_general",
    "distance_to_nearest_own_structure",
    "legal_build_sites",
    "actor_value",
    "total_build_probability",
    "best_build_probability",
    "control_probability",
    "best_build_logit_margin",
    "best_build_rank",
)

HIDDEN_GEOMETRY_FEATURES = ("distance_to_enemy_land_true",)


def _bootstrap_group_ci(values, groups, seed, draws=10_000):
    unique = np.unique(groups)
    grouped = [values[groups == group] for group in unique]
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sample = rng.integers(0, len(grouped), size=len(grouped))
        selected = [grouped[index] for index in sample]
        estimates[draw] = np.concatenate(selected).mean()
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def _spearman(x, y):
    x_rank = np.argsort(np.argsort(x)).astype(np.float64)
    y_rank = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _make_estimator(name, seed):
    if name == "ridge":
        return make_pipeline(
            SimpleImputer(), StandardScaler(), Ridge(alpha=10.0)
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=2.0,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=20,
            max_features=0.8,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(name)


def _cross_fit(features, target, groups, estimator_name, seed):
    predictions = np.empty(len(target), dtype=np.float64)
    splitter = GroupKFold(n_splits=5)
    for fold, (train, test) in enumerate(splitter.split(features, target, groups)):
        estimator = _make_estimator(estimator_name, seed + fold)
        estimator.fit(features[train], target[train])
        predictions[test] = estimator.predict(features[test])
    return predictions


def _selector_summary(prediction, test_delta, test_control, groups, seed):
    selected = prediction > 0
    uplift_by_state = selected * test_delta
    selected_count = int(selected.sum())
    return {
        "selection_rate": float(selected.mean()),
        "selected_states": selected_count,
        "policy_score": float(np.mean(test_control + uplift_by_state)),
        "control_score": float(np.mean(test_control)),
        "overall_uplift": float(uplift_by_state.mean()),
        "source_game_cluster_bootstrap_95": _bootstrap_group_ci(
            uplift_by_state, groups, seed
        ),
        "conditional_build_effect": (
            float(test_delta[selected].mean()) if selected_count else None
        ),
        "prediction_vs_test_delta_spearman": _spearman(prediction, test_delta),
    }


def _symmetric_cross_fit(
    features,
    delta_first,
    delta_second,
    control_first,
    control_second,
    groups,
    estimator_name,
    seed,
):
    first_prediction = _cross_fit(
        features, delta_first, groups, estimator_name, seed
    )
    second_prediction = _cross_fit(
        features, delta_second, groups, estimator_name, seed + 100
    )
    first_to_second = _selector_summary(
        first_prediction, delta_second, control_second, groups, seed + 200
    )
    second_to_first = _selector_summary(
        second_prediction, delta_first, control_first, groups, seed + 300
    )
    selected_first = first_prediction > 0
    selected_second = second_prediction > 0
    symmetric_uplift = 0.5 * (
        selected_first * delta_second + selected_second * delta_first
    )
    return {
        "first_half_train_second_half_test": first_to_second,
        "second_half_train_first_half_test": second_to_first,
        "symmetric": {
            "selection_rate": float(
                0.5 * (selected_first.mean() + selected_second.mean())
            ),
            "overall_uplift": float(symmetric_uplift.mean()),
            "source_game_cluster_bootstrap_95": _bootstrap_group_ci(
                symmetric_uplift, groups, seed + 400
            ),
            "conditional_build_effect": float(
                (
                    (selected_first * delta_second).sum()
                    + (selected_second * delta_first).sum()
                )
                / max(selected_first.sum() + selected_second.sum(), 1)
            ),
            "prediction_vs_test_delta_spearman": float(
                0.5
                * (
                    _spearman(first_prediction, delta_second)
                    + _spearman(second_prediction, delta_first)
                )
            ),
        },
    }


def _fixed_rule_summary(mask, delta_first, delta_second, control_first, control_second, groups, seed):
    uplift = mask * 0.5 * (delta_first + delta_second)
    return {
        "selection_rate": float(mask.mean()),
        "overall_uplift": float(uplift.mean()),
        "source_game_cluster_bootstrap_95": _bootstrap_group_ci(
            uplift, groups, seed
        ),
        "conditional_build_effect": float(
            (0.5 * (delta_first + delta_second))[mask].mean()
        ),
        "policy_score": float(
            np.mean(0.5 * (control_first + control_second) + uplift)
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def main():
    args = parse_args()
    raw = np.load(args.atlas)
    outcome = raw["result__outcome"]
    if outcome.shape[1] % 2:
        raise ValueError("Expected an even number of paired repetitions")
    half = outcome.shape[1] // 2
    control = outcome[:, :, 0]
    build = outcome[:, :, 1]
    delta_first = (build[:, :half] - control[:, :half]).mean(axis=1)
    delta_second = (build[:, half:] - control[:, half:]).mean(axis=1)
    control_first = control[:, :half].mean(axis=1)
    control_second = control[:, half:].mean(axis=1)
    source_batch = raw["metadata__source_batch"].astype(np.int64)
    source_slot = raw["metadata__source_slot"].astype(np.int64)
    source_game = source_batch * 512 + source_slot // 12

    features = {
        name: raw[f"feature__{name}"].astype(np.float64)
        for name in (*OBSERVABLE_FEATURES, *HIDDEN_GEOMETRY_FEATURES)
    }
    # Log probabilities are numerically better behaved than raw probabilities.
    for name in ("total_build_probability", "best_build_probability"):
        features[name] = np.log10(np.maximum(features[name], 1e-30))

    feature_sets = {
        "observable": OBSERVABLE_FEATURES,
        "observable_plus_true_enemy_distance": (
            *OBSERVABLE_FEATURES,
            *HIDDEN_GEOMETRY_FEATURES,
        ),
    }
    learned = {}
    for set_name, feature_names in feature_sets.items():
        matrix = np.column_stack([features[name] for name in feature_names])
        learned[set_name] = {
            estimator_name: _symmetric_cross_fit(
                matrix,
                delta_first,
                delta_second,
                control_first,
                control_second,
                source_game,
                estimator_name,
                args.seed + index * 1_000,
            )
            for index, estimator_name in enumerate(
                ("ridge", "hist_gradient_boosting", "random_forest")
            )
        }

    army_margin = features["army_margin"]
    land_margin = features["land_margin"]
    post_build_garrison = features["post_build_garrison"]
    fixed_rules = {
        "land_margin_le_-10": land_margin <= -10,
        "army_margin_negative": army_margin < 0,
        "army_and_land_margin_negative": (army_margin < 0) & (land_margin < 0),
        "land_margin_le_-10_and_garrison_ge_10": (
            (land_margin <= -10) & (post_build_garrison >= 10)
        ),
    }
    rule_results = {
        name: _fixed_rule_summary(
            mask,
            delta_first,
            delta_second,
            control_first,
            control_second,
            source_game,
            args.seed + 10_000 + index,
        )
        for index, (name, mask) in enumerate(fixed_rules.items())
    }

    all_delta = 0.5 * (delta_first + delta_second)
    report = {
        "states": int(len(all_delta)),
        "unique_source_games": int(len(np.unique(source_game))),
        "protocol": (
            "5-fold GroupKFold by source game; train on one rollout half, "
            "test on the other, then swap halves"
        ),
        "forced_build_everywhere_delta": float(all_delta.mean()),
        "learned_selectors": learned,
        "descriptive_fixed_rules": rule_results,
        "feature_sets": {
            name: list(feature_names) for name, feature_names in feature_sets.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Minimal four-device correctness preflight for castle-counterfactual PPO."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tools.export_competition_checkpoint import _extract_weights

from generals.core.game import get_observation

from .actions import legal_action_mask
from .config import TrainingConfig
from .observation import augment_observation, init_observation_memory, temporal_input
from .train import (
    _attach_zero_build_difference_head,
    _learning_rate,
    _load_checkpoint_state,
    _migrate_optimizer_state_with_zero_head,
    _sha256_file,
    build_network,
    make_environment,
    train,
)


def _load_historical_and_migrate(config: TrainingConfig, checkpoint: Path):
    historical = replace(
        config,
        counterfactual_castle_training=False,
        residual_build_kind_head=False,
    )
    key = jax.random.PRNGKey(config.seed)
    key, network_key = jax.random.split(key)
    old_network = build_network(historical, network_key)
    new_network = build_network(config, network_key)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(partial(_learning_rate, config)),
    )
    old_optimizer = optimizer.init(eqx.filter(old_network, eqx.is_inexact_array))
    new_optimizer = optimizer.init(eqx.filter(new_network, eqx.is_inexact_array))
    skeleton = (
        old_network,
        old_optimizer,
        old_network,
        jnp.int32(0),
        jnp.int32(0),
        key,
    )
    old_raw, old_opt, old_ema, iteration, stage, loaded_key = _load_checkpoint_state(
        checkpoint, skeleton, historical
    )
    migrated_raw = _attach_zero_build_difference_head(old_raw, new_network)
    migrated_ema = _attach_zero_build_difference_head(old_ema, new_network)
    migrated_opt = _migrate_optimizer_state_with_zero_head(old_opt, new_optimizer)
    return (
        old_raw,
        old_ema,
        migrated_raw,
        migrated_ema,
        migrated_opt,
        int(iteration),
        int(stage),
        loaded_key,
    )


def _assert_forward_identity(config, old_network, migrated_network):
    # Variable 18--21 boards comprise 16 (height, width) combinations, so the
    # pool must contain at least one board per combination.
    environment = make_environment(config, config.curriculum[-1], pool_size=16)
    pool, _ = environment.reset(jax.random.PRNGKey(991))
    state = jax.tree.map(lambda value: value[0], pool)
    observation = get_observation(state, 0)
    memory = init_observation_memory(
        config.pad_to, config.history_size, config.temporal_window
    )
    augmented, memory = augment_observation(
        observation,
        memory,
        state.board_mask,
        config.observation_schema,
        environment.deathtouch_turn or 800,
    )
    history = temporal_input(memory)
    mask = legal_action_mask(observation, state.board_mask)
    old_outputs = old_network.forward(augmented, history, mask)
    migrated_outputs = migrated_network.forward(augmented, history, mask)
    for old, migrated in zip(old_outputs, migrated_outputs, strict=True):
        if not np.array_equal(np.asarray(old), np.asarray(migrated)):
            raise AssertionError("Auxiliary-head migration changed actor/critic output")
    difference = migrated_network.build_difference(augmented, history, mask)
    legal = np.asarray(mask[8 * 21 * 21 : 9 * 21 * 21]).reshape(21, 21)
    values = np.asarray(difference)
    if not np.all(values[legal] == 0):
        raise AssertionError("Zero-initialized build-difference head is nonzero")
    if not np.all(np.isnan(values[~legal])):
        raise AssertionError("Illegal build-difference cells are not masked")
    if migrated_network.build_kind_head is None:
        raise AssertionError("Migrated network has no residual build-kind head")
    if not np.all(np.asarray(migrated_network.build_kind_head.weight) == 0):
        raise AssertionError("Zero-initialized build-kind head has nonzero weights")
    if not np.all(np.asarray(migrated_network.build_kind_head.bias) == 0):
        raise AssertionError("Zero-initialized build-kind head has nonzero bias")


def _assert_actor_preference_signs(temperature: float, weight_scale: float):
    def loss(margin, delta):
        target = jax.nn.sigmoid(delta / temperature)
        magnitude = jnp.clip(jnp.abs(delta) / weight_scale, 0.0, 1.0)
        return magnitude * -(
            target * jax.nn.log_sigmoid(margin)
            + (1.0 - target) * jax.nn.log_sigmoid(-margin)
        )

    positive_gradient = float(jax.grad(loss)(jnp.asarray(0.0), jnp.asarray(1.0)))
    negative_gradient = float(jax.grad(loss)(jnp.asarray(0.0), jnp.asarray(-1.0)))
    uncertain_gradient = float(jax.grad(loss)(jnp.asarray(0.0), jnp.asarray(0.0)))
    if not positive_gradient < 0:
        raise AssertionError("Positive causal labels do not raise build margins")
    if not negative_gradient > 0:
        raise AssertionError("Negative causal labels do not lower build margins")
    if uncertain_gradient != 0:
        raise AssertionError("Zero-weight uncertain actor example has a gradient")


def run(args: argparse.Namespace) -> dict:
    production = TrainingConfig.from_toml(args.config)
    if not production.counterfactual_castle_training:
        raise ValueError("Preflight requires the treatment config")
    if not production.residual_build_kind_head:
        raise ValueError("Preflight requires the residual build-kind head")
    if jax.device_count() != 4:
        raise RuntimeError(f"Expected exactly four visible H100s, got {jax.devices()}")
    if _sha256_file(args.resume) != production.resume_checkpoint_sha256:
        raise ValueError("Historical resume checkpoint hash mismatch")
    (
        old_raw,
        old_ema,
        migrated_raw,
        migrated_ema,
        migrated_opt,
        iteration,
        stage,
        _,
    ) = _load_historical_and_migrate(production, args.resume)
    if iteration != production.parent_final_iteration or stage != production.resume_start_stage:
        raise AssertionError("Historical checkpoint lineage mismatch")
    _assert_forward_identity(production, old_raw, migrated_raw)
    _assert_forward_identity(production, old_ema, migrated_ema)
    _assert_actor_preference_signs(
        production.counterfactual_actor_temperature,
        production.counterfactual_actor_weight_scale,
    )
    migrated_adam = (
        migrated_opt[1][0]
        if isinstance(migrated_opt[1], tuple)
        else migrated_opt[1]
    )
    if not np.all(np.asarray(migrated_adam.mu.build_difference_head.weight) == 0):
        raise AssertionError("Migrated Adam first moment for auxiliary head is nonzero")
    if not np.all(np.asarray(migrated_adam.nu.build_difference_head.weight) == 0):
        raise AssertionError("Migrated Adam second moment for auxiliary head is nonzero")
    if not np.all(np.asarray(migrated_adam.mu.build_kind_head.weight) == 0):
        raise AssertionError("Migrated Adam first moment for build-kind head is nonzero")
    if not np.all(np.asarray(migrated_adam.nu.build_kind_head.weight) == 0):
        raise AssertionError("Migrated Adam second moment for build-kind head is nonzero")
    del old_raw, old_ema, migrated_raw, migrated_ema, migrated_opt
    gc.collect()

    smoke = replace(
        production,
        run_name="castle_counterfactual_preflight",
        output_dir=str(args.output_dir),
        wandb_project=None,
        wandb_entity=None,
        wandb_group=None,
        wandb_run_id=None,
        wandb_run_name=None,
        wandb_tags=(),
        num_envs=8,
        num_steps=16,
        num_iterations=production.parent_final_iteration + 1,
        minibatch_size=16,
        pool_size=512,
        reset_pool_every=0,
        eval_every=0,
        eval_games=16,
        checkpoint_every=1,
        latest_checkpoint_every=1,
        league_eval_after_training=False,
        league_eval_every=0,
        league_opponents=(),
        league_checkpoint_name=None,
        league_checkpoint_path=None,
        league_checkpoint_sha256=None,
        counterfactual_source_states=32,
        counterfactual_source_games_per_device=32,
        counterfactual_uniform_source_states=16,
        counterfactual_promising_source_states=8,
        counterfactual_hard_source_states=8,
        counterfactual_repetitions=2,
        counterfactual_buffer_capacity=128,
        counterfactual_actor_minibatch_size_per_device=4,
        counterfactual_successor_minibatch_size_per_device=4,
        counterfactual_unique_examples_full_weight=32,
        counterfactual_actor_uniform_per_device=2,
        counterfactual_actor_positive_per_device=1,
        counterfactual_actor_negative_per_device=1,
    )
    final_raw, _, final_ema = train(smoke, resume=str(args.resume))
    smoke_run = smoke.run_dir
    terminal = smoke_run / "terminal.eqx"
    if not terminal.is_file() or not (smoke_run / "terminal.counterfactual.json").is_file():
        raise AssertionError("Treatment smoke did not checkpoint full state")
    metrics = [
        json.loads(line)
        for line in (smoke_run / "metrics.jsonl").read_text().splitlines()
    ]
    training = [record for record in metrics if "loss" in record]
    if len(training) != 1:
        raise AssertionError(f"Expected one treatment update, found {len(training)}")
    required = (
        "counterfactual_actor_loss",
        "counterfactual_successor_value_loss",
        "counterfactual_delta_loss",
        "counterfactual_build_kind_residual_positive",
        "counterfactual_build_kind_residual_negative",
        "counterfactual_conditional_build_probability_positive",
        "counterfactual_conditional_build_rank_positive",
        "diagnostic_build_kind_actor_gradient_norm",
        "diagnostic_cf_to_ppo_gradient_ratio",
        "approximate_kl",
        "ppo_castle/count/builds",
    )
    if any(name not in training[0] for name in required):
        raise AssertionError("Treatment metrics are incomplete")
    if not all(np.isfinite(training[0][name]) for name in required):
        raise AssertionError("Treatment update emitted non-finite metrics")
    if training[0]["diagnostic_build_kind_actor_gradient_norm"] <= 0:
        raise AssertionError("Counterfactual actor loss did not reach build-kind head")

    # Deserialize the newly versioned checkpoint, including head/optimizer/EMA.
    key = jax.random.PRNGKey(smoke.seed)
    key, network_key = jax.random.split(key)
    skeleton_network = build_network(smoke, network_key)
    optimizer = optax.chain(
        optax.clip_by_global_norm(smoke.max_grad_norm),
        optax.adam(partial(_learning_rate, smoke)),
    )
    skeleton_optimizer = optimizer.init(
        eqx.filter(skeleton_network, eqx.is_inexact_array)
    )
    restored = _load_checkpoint_state(
        terminal,
        (
            skeleton_network,
            skeleton_optimizer,
            skeleton_network,
            jnp.int32(0),
            jnp.int32(0),
            key,
        ),
        smoke,
    )
    if int(restored[3]) != smoke.num_iterations:
        raise AssertionError("New checkpoint iteration did not restore")
    if restored[0].build_difference_head is None or restored[2].build_difference_head is None:
        raise AssertionError("Auxiliary head did not restore in raw and EMA trees")
    if restored[0].build_kind_head is None or restored[2].build_kind_head is None:
        raise AssertionError("Build-kind head did not restore in raw and EMA trees")
    exported = _extract_weights(restored[0])
    for name in ("build_kind_head.weight", "build_kind_head.bias"):
        if name not in exported:
            raise AssertionError(f"Competition export omitted {name}")
    report = {
        "status": "passed",
        "devices": [str(device) for device in jax.devices()],
        "historical_iteration": iteration,
        "historical_actor_critic_exact": True,
        "zero_head_exact": True,
        "zero_build_kind_head_exact": True,
        "optimizer_migration": True,
        "actor_preference_gradient_signs": True,
        "treatment_update_finite": True,
        "checkpoint_resume_complete": True,
        "competition_export_includes_build_kind_head": True,
        "training_metrics": training[0],
        "terminal_checkpoint": str(terminal),
        "raw_parameter_leaves": len(jax.tree.leaves(eqx.filter(final_raw, eqx.is_array))),
        "ema_parameter_leaves": len(jax.tree.leaves(eqx.filter(final_ema, eqx.is_array))),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "generals/training/configs/castle_counterfactual_treatment_from_4000.toml"
        ),
    )
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/preflight")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/preflight/castle_counterfactual_preflight.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

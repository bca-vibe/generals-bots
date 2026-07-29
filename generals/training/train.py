"""Command-line entry point for competition baseline pre-training."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from generals.core.env import GeneralsEnv

from .config import CurriculumStage, TrainingConfig
from .evaluation import evaluate_paired_vs_random
from .model import CompetitionTransformer
from .observation import init_observation_memory
from .ppo import compute_gae, ppo_epoch
from .rollout import collect_self_play_rollout


def make_environment(
    config: TrainingConfig, stage: CurriculumStage, *, pool_size: int | None = None
) -> GeneralsEnv:
    """Construct an environment with invariant competition rules and one curriculum stage."""
    return GeneralsEnv(
        min_grid_size=config.min_grid_size,
        max_grid_size=config.max_grid_size,
        pad_to=config.pad_to,
        truncation=config.truncation,
        mountain_density_range=(
            config.mountain_density_min,
            config.mountain_density_max,
        ),
        # Match the official generator exactly: it first converts 9-11 sampled
        # mountains to neutral castles, then build_castles strips those castles
        # to plain ground before play begins.
        num_castles_range=(9, 11),
        min_generals_distance=stage.min_generals_distance,
        max_generals_distance=stage.max_generals_distance,
        pool_size=pool_size or config.pool_size,
        castle_val_range=(20, 26),
        perfect_info=False,
        build_castles=True,
        deathtouch_turn=config.deathtouch_turn,
    )


def build_network(config: TrainingConfig, key) -> CompetitionTransformer:
    return CompetitionTransformer(
        board_size=config.pad_to,
        input_channels=24 + 2 * config.history_size,
        history_window=config.temporal_window,
        patch_size=config.patch_size,
        depth=config.depth,
        model_dim=config.embed_dim,
        heads=config.attention_heads,
        ff_factor=config.ff_factor,
        value_bins=config.value_bins,
        value_min=config.value_min,
        value_max=config.value_max,
        use_bf16=config.use_bf16,
        key=key,
    )


def _batched_memory(config: TrainingConfig):
    memory = init_observation_memory(
        config.pad_to, config.history_size, config.temporal_window
    )
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (config.num_envs, *value.shape)), memory
    )


def _replicate_for_pmap(tree):
    """Add a leading device axis; pmap shards that axis across accelerators."""
    device_count = jax.device_count()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (device_count, *value.shape)), tree
    )


def _learning_rate(config: TrainingConfig, optimizer_step):
    kept_samples = int(
        2 * config.num_envs * config.num_steps * config.advantage_top_fraction
    )
    optimizer_steps_per_iteration = (
        config.ppo_epochs * kept_samples // config.minibatch_size
    )
    iteration = optimizer_step / optimizer_steps_per_iteration + 1.0
    raw = config.learning_rate_numerator / iteration**config.learning_rate_exponent
    return jnp.clip(raw, config.learning_rate_min, config.learning_rate_max)


def _entropy_coefficient(config: TrainingConfig, iteration: int) -> float:
    return max(
        config.entropy_start / (iteration + 1) ** config.entropy_power,
        config.entropy_min,
    )


def _write_metrics(path: Path, metrics: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")


def _save_checkpoint(
    path: Path,
    network,
    optimizer_state,
    ema_network,
    iteration: int,
    stage_index: int,
    key,
):
    eqx.tree_serialise_leaves(
        path,
        (
            network,
            optimizer_state,
            ema_network,
            jnp.int32(iteration),
            jnp.int32(stage_index),
            key,
        ),
    )


def _make_evaluator(config: TrainingConfig, environment, n_maps: int, truncation: int):
    return eqx.filter_jit(
        lambda pool, network, key: evaluate_paired_vs_random(
            environment,
            pool,
            network,
            key,
            n_maps,
            truncation,
            pad_to=config.pad_to,
            history_size=config.history_size,
            temporal_window=config.temporal_window,
        )
    )


def train(config: TrainingConfig, *, resume: str | None = None):
    """Train the shared AverageJoe-style baseline and return its final state."""
    config.validate()
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    metrics_path = run_dir / "metrics.jsonl"

    print(f"Devices ({jax.device_count()}): {jax.devices()}")
    print(
        f"Competition baseline: L={config.depth}, d={config.embed_dim}, "
        f"heads={config.attention_heads}, envs/device={config.num_envs}"
    )

    key = jax.random.PRNGKey(config.seed)
    key, network_key = jax.random.split(key)
    network = build_network(config, network_key)
    schedule = partial(_learning_rate, config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm), optax.adam(schedule)
    )
    optimizer_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    ema_network = network
    start_iteration = 0
    stage_index = 0
    if resume:
        skeleton = (
            network,
            optimizer_state,
            ema_network,
            jnp.int32(0),
            jnp.int32(0),
            key,
        )
        (
            network,
            optimizer_state,
            ema_network,
            saved_iteration,
            saved_stage_index,
            key,
        ) = eqx.tree_deserialise_leaves(resume, skeleton)
        start_iteration = int(saved_iteration)
        stage_index = int(saved_stage_index)
        if stage_index >= len(config.curriculum):
            raise ValueError(
                f"Checkpoint curriculum stage {stage_index} is not present in the config"
            )
        print(
            f"Resumed {resume} at iteration {start_iteration}, "
            f"curriculum stage {stage_index}"
        )

    trainable = eqx.filter(network, eqx.is_inexact_array)
    parameter_count = sum(value.size for value in jax.tree.leaves(trainable))
    print(f"Array parameters: {parameter_count:,}")

    stage = config.curriculum[stage_index]
    environment = make_environment(config, stage)
    key, pool_key = jax.random.split(key)
    pool, _ = environment.reset(pool_key)
    pool_replicated = _replicate_for_pmap(pool)

    eval_pool_size = max(config.eval_games // 2, 16)
    evaluation_environment = make_environment(config, stage, pool_size=eval_pool_size)
    key, evaluation_pool_key = jax.random.split(key)
    evaluation_pool, _ = evaluation_environment.reset(evaluation_pool_key)
    evaluator = _make_evaluator(
        config, evaluation_environment, config.eval_games // 2, config.truncation
    )

    parameters, static = eqx.partition(network, eqx.is_inexact_array)
    parameters = _replicate_for_pmap(parameters)
    optimizer_state = _replicate_for_pmap(optimizer_state)
    ema_parameters, _ = eqx.partition(ema_network, eqx.is_inexact_array)

    def sample_initial_states(pool_shard, rng):
        indices = jax.random.randint(
            rng, (config.num_envs,), 0, environment.pool_size
        )
        sampled = jax.tree.map(lambda value: value[indices], pool_shard)
        return sampled._replace(pool_idx=indices.astype(jnp.int32))

    sample_initial_states_pmapped = jax.pmap(sample_initial_states)
    key, states_key = jax.random.split(key)
    device_keys = jax.random.split(states_key, jax.device_count())
    states = sample_initial_states_pmapped(pool_replicated, device_keys)

    per_device_memory = _batched_memory(config)
    memory_zero = _replicate_for_pmap(per_device_memory)
    memory_one = _replicate_for_pmap(per_device_memory)
    key, rollout_key = jax.random.split(key)
    rollout_keys = jax.random.split(rollout_key, jax.device_count())

    def rollout_shard(params, states, rng, mem_zero, mem_one, pool_shard):
        shard_network = eqx.combine(params, static)
        return collect_self_play_rollout(
            states,
            pool_shard,
            environment,
            shard_network,
            rng,
            mem_zero,
            mem_one,
            config.num_steps,
        )

    rollout_pmapped = jax.pmap(rollout_shard)

    gae_pmapped = jax.pmap(
        lambda rewards, values, next_values, terminated, truncated: compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            config.gamma,
            config.gae_lambda,
        )
    )

    @partial(jax.pmap, axis_name="devices")
    def normalize_advantages(advantages):
        mean = jax.lax.pmean(advantages.mean(), axis_name="devices")
        mean_square = jax.lax.pmean((advantages**2).mean(), axis_name="devices")
        standard_deviation = jnp.sqrt(jnp.maximum(mean_square - mean**2, 0.0))
        return (advantages - mean) / (standard_deviation + 1e-8)

    per_device_samples = 2 * config.num_envs * config.num_steps
    kept_samples = int(per_device_samples * config.advantage_top_fraction)

    @jax.pmap
    def select_samples(advantages):
        _, indices = jax.lax.top_k(jnp.abs(advantages.reshape(-1)), kept_samples)
        return indices

    @partial(jax.pmap, axis_name="devices")
    def update_shard(params, opt_state, batch, indices, rng, entropy_coefficient):
        shard_network = eqx.combine(params, static)
        shard_network, opt_state, metrics = ppo_epoch(
            shard_network,
            opt_state,
            batch,
            indices,
            optimizer,
            rng,
            minibatch_size=config.minibatch_size,
            clip_epsilon=config.clip_epsilon,
            value_coefficient=config.value_coefficient,
            entropy_coefficient=entropy_coefficient,
            value_bins=config.value_bins,
            value_min=config.value_min,
            value_max=config.value_max,
            hl_gauss_sigma=config.hl_gauss_sigma,
            axis_name="devices",
        )
        params, _ = eqx.partition(shard_network, eqx.is_inexact_array)
        metrics = jax.lax.pmean(metrics, axis_name="devices")
        return params, opt_state, metrics

    train_started = time.perf_counter()
    for iteration in range(start_iteration, config.num_iterations):
        iteration_started = time.perf_counter()
        rollout_started = time.perf_counter()
        states, rollout, rollout_keys, memory_zero, memory_one = rollout_pmapped(
            parameters,
            states,
            rollout_keys,
            memory_zero,
            memory_one,
            pool_replicated,
        )
        jax.block_until_ready(states)
        rollout_seconds = time.perf_counter() - rollout_started
        (
            observations,
            histories,
            masks,
            actions,
            old_log_probs,
            values,
            next_values,
            rewards,
            terminated,
            truncated,
            winners,
        ) = rollout

        advantages = gae_pmapped(
            rewards, values, next_values, terminated, truncated
        )
        returns = advantages + values
        raw_advantage_std = float(advantages.std())
        advantages = normalize_advantages(advantages)
        sample_indices = select_samples(advantages)
        batch = (
            observations,
            histories,
            masks,
            actions,
            old_log_probs,
            advantages,
            returns,
        )

        update_started = time.perf_counter()
        entropy_coefficient = _entropy_coefficient(config, iteration)
        entropy_by_device = jnp.full(
            (jax.device_count(),), entropy_coefficient, dtype=jnp.float32
        )
        epochs_used = 0
        metrics = None
        for _ in range(config.ppo_epochs):
            split_keys = jax.vmap(jax.random.split)(rollout_keys)
            rollout_keys, update_keys = split_keys[:, 0], split_keys[:, 1]
            parameters, optimizer_state, metrics = update_shard(
                parameters,
                optimizer_state,
                batch,
                sample_indices,
                update_keys,
                entropy_by_device,
            )
            epochs_used += 1
            if float(metrics["approximate_kl"][0]) > config.target_kl:
                break
        jax.block_until_ready(parameters)
        update_seconds = time.perf_counter() - update_started

        current_parameters = jax.tree.map(lambda value: value[0], parameters)
        ema_parameters = jax.tree.map(
            lambda ema, current: config.ema_decay * ema
            + (1.0 - config.ema_decay) * current,
            ema_parameters,
            current_parameters,
        )

        done = terminated | truncated
        p0_done = done[:, :, : config.num_envs]
        p0_terminated = terminated[:, :, : config.num_envs]
        p0_winners = winners[:, :, : config.num_envs]
        episodes = int(p0_done.sum())
        wins = int(jnp.sum(p0_terminated & (p0_winners == 0)))
        losses = int(jnp.sum(p0_terminated & (p0_winners == 1)))
        draws = episodes - wins - losses
        return_variance = float(jnp.var(returns))
        explained_variance = 1.0 - float(jnp.var(returns - values)) / max(
            return_variance, 1e-8
        )
        iteration_seconds = time.perf_counter() - iteration_started
        total_samples = (
            jax.device_count() * 2 * config.num_envs * config.num_steps
        )
        samples_per_second = total_samples / iteration_seconds
        device_metrics = jax.tree.map(lambda value: float(value[0]), metrics)
        current_learning_rate = float(_learning_rate(config, iteration * (
            config.ppo_epochs * kept_samples // config.minibatch_size
        )))
        record = {
            "iteration": iteration + 1,
            "wall_seconds": time.perf_counter() - train_started,
            "stage": stage_index,
            "samples_per_second": samples_per_second,
            "rollout_seconds": rollout_seconds,
            "update_seconds": update_seconds,
            "episodes": episodes,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "mean_reward": float(rewards.mean()),
            "explained_variance": explained_variance,
            "raw_advantage_std": raw_advantage_std,
            "learning_rate": current_learning_rate,
            "entropy_coefficient": entropy_coefficient,
            "epochs_used": epochs_used,
            **device_metrics,
        }

        if (iteration + 1) % config.metrics_every == 0:
            _write_metrics(metrics_path, record)
            print(
                f"iter {iteration + 1:6d} | loss {record['loss']:.4f} | "
                f"entropy {record['entropy']:.3f} | KL {record['approximate_kl']:.4f} | "
                f"episodes {episodes} W/L/D {wins}/{losses}/{draws} | "
                f"EV {explained_variance:.2f} | {samples_per_second:,.0f} samples/s"
            )

        should_evaluate = config.eval_every > 0 and (
            (iteration + 1) % config.eval_every == 0
        )
        if should_evaluate:
            key, evaluation_key = jax.random.split(key)
            ema_network = eqx.combine(ema_parameters, static)
            evaluation, _ = evaluator(evaluation_pool, ema_network, evaluation_key)
            evaluation = jax.tree.map(float, evaluation)
            evaluation_record = {
                "iteration": iteration + 1,
                **{f"evaluation/{name}": value for name, value in evaluation.items()},
            }
            _write_metrics(metrics_path, evaluation_record)
            print(
                f"  eval EMA: {int(evaluation['wins'])}W/"
                f"{int(evaluation['losses'])}L/{int(evaluation['draws'])}D, "
                f"score={evaluation['score']:.3f}"
            )

            if stage_index + 1 < len(config.curriculum):
                next_stage = config.curriculum[stage_index + 1]
                if evaluation["score"] >= stage.advance_win_rate:
                    stage_index += 1
                    stage = next_stage
                    print(
                        f"  curriculum -> stage {stage_index}: general distance "
                        f"{stage.min_generals_distance}-{stage.max_generals_distance}"
                    )
                    environment = make_environment(config, stage)
                    key, pool_key = jax.random.split(key)
                    pool, _ = environment.reset(pool_key)
                    pool_replicated = _replicate_for_pmap(pool)
                    rollout_pmapped = jax.pmap(rollout_shard)
                    key, states_key = jax.random.split(key)
                    device_keys = jax.random.split(states_key, jax.device_count())
                    states = sample_initial_states_pmapped(pool_replicated, device_keys)
                    memory_zero = _replicate_for_pmap(per_device_memory)
                    memory_one = _replicate_for_pmap(per_device_memory)

                    evaluation_environment = make_environment(
                        config, stage, pool_size=eval_pool_size
                    )
                    key, evaluation_pool_key = jax.random.split(key)
                    evaluation_pool, _ = evaluation_environment.reset(evaluation_pool_key)
                    evaluator = _make_evaluator(
                        config,
                        evaluation_environment,
                        config.eval_games // 2,
                        config.truncation,
                    )

        if (
            config.reset_pool_every > 0
            and (iteration + 1) % config.reset_pool_every == 0
        ):
            key, pool_key = jax.random.split(key)
            pool, _ = environment.reset(pool_key)
            pool_replicated = _replicate_for_pmap(pool)

        if (iteration + 1) % config.checkpoint_every == 0:
            current_network = eqx.combine(current_parameters, static)
            current_optimizer_state = jax.tree.map(
                lambda value: value[0], optimizer_state
            )
            ema_network = eqx.combine(ema_parameters, static)
            checkpoint = run_dir / f"checkpoint_{iteration + 1:06d}.eqx"
            _save_checkpoint(
                checkpoint,
                current_network,
                current_optimizer_state,
                ema_network,
                iteration + 1,
                stage_index,
                key,
            )
            _save_checkpoint(
                run_dir / "latest.eqx",
                current_network,
                current_optimizer_state,
                ema_network,
                iteration + 1,
                stage_index,
                key,
            )
            print(f"  saved {checkpoint}")

        del rollout, batch, observations, histories, masks, actions, old_log_probs
        del values, next_values, rewards, terminated, truncated, winners
        del advantages, returns, sample_indices

    final_network = eqx.combine(
        jax.tree.map(lambda value: value[0], parameters), static
    )
    final_optimizer_state = jax.tree.map(lambda value: value[0], optimizer_state)
    final_ema_network = eqx.combine(ema_parameters, static)
    _save_checkpoint(
        run_dir / "final.eqx",
        final_network,
        final_optimizer_state,
        final_ema_network,
        config.num_iterations,
        stage_index,
        key,
    )
    return final_network, final_optimizer_state, final_ema_network


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="generals/training/configs/competition_l7.toml",
        help="Path to a TOML TrainingConfig",
    )
    parser.add_argument("--resume", help="Path to a checkpoint produced by this trainer")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig.from_toml(args.config)
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()

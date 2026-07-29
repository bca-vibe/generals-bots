"""Command-line entry point for competition baseline pre-training."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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


def _learning_rate_float(config: TrainingConfig, optimizer_step: int) -> float:
    """Same schedule as _learning_rate, in plain Python for logging."""
    kept_samples = int(
        2 * config.num_envs * config.num_steps * config.advantage_top_fraction
    )
    optimizer_steps_per_iteration = (
        config.ppo_epochs * kept_samples // config.minibatch_size
    )
    iteration = optimizer_step / optimizer_steps_per_iteration + 1.0
    raw = config.learning_rate_numerator / iteration**config.learning_rate_exponent
    return min(max(raw, config.learning_rate_min), config.learning_rate_max)


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


def make_prepare_batch(config: TrainingConfig):
    """GAE -> returns -> normalization -> top-k selection -> prep metrics,
    fused into one compiled dispatch (was five dispatches plus eager glue)."""
    kept_samples = int(
        2 * config.num_envs * config.num_steps * config.advantage_top_fraction
    )

    @partial(jax.pmap, axis_name="devices")
    def prepare_batch(rewards, values, next_values, terminated, truncated, winners):
        advantages = compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            config.gamma,
            config.gae_lambda,
        )
        returns = advantages + values

        mean = jax.lax.pmean(advantages.mean(), axis_name="devices")
        mean_square = jax.lax.pmean((advantages**2).mean(), axis_name="devices")
        raw_std = jnp.sqrt(jnp.maximum(mean_square - mean**2, 0.0))
        normalized = (advantages - mean) / (raw_std + 1e-8)
        _, indices = jax.lax.top_k(jnp.abs(normalized.reshape(-1)), kept_samples)

        # Episode outcome counts for player-0 seats, summed across devices;
        # shards are equal-sized so pmean of per-shard moments equals the
        # global moments the host loop previously computed eagerly.
        done = terminated | truncated
        p0_done = done[:, : config.num_envs]
        p0_terminated = terminated[:, : config.num_envs]
        p0_winners = winners[:, : config.num_envs]
        episodes = jax.lax.psum(p0_done.sum(), axis_name="devices")
        wins = jax.lax.psum(
            jnp.sum(p0_terminated & (p0_winners == 0)), axis_name="devices"
        )
        losses = jax.lax.psum(
            jnp.sum(p0_terminated & (p0_winners == 1)), axis_name="devices"
        )

        returns_mean = jax.lax.pmean(returns.mean(), axis_name="devices")
        returns_square = jax.lax.pmean((returns**2).mean(), axis_name="devices")
        return_variance = returns_square - returns_mean**2
        residual = returns - values
        residual_mean = jax.lax.pmean(residual.mean(), axis_name="devices")
        residual_square = jax.lax.pmean((residual**2).mean(), axis_name="devices")
        residual_variance = residual_square - residual_mean**2
        explained_variance = 1.0 - residual_variance / jnp.maximum(
            return_variance, 1e-8
        )

        prep_metrics = {
            "raw_advantage_std": raw_std,
            "mean_reward": jax.lax.pmean(rewards.mean(), axis_name="devices"),
            "episodes": episodes,
            "wins": wins,
            "losses": losses,
            "explained_variance": explained_variance,
        }
        return normalized, returns, indices, prep_metrics

    return prepare_batch


def make_ema_step(config: TrainingConfig):
    """One compiled dispatch updating the replicated EMA tree in place of
    eager per-leaf host operations."""

    @jax.pmap
    def ema_step(ema, current):
        return jax.tree.map(
            lambda e, c: config.ema_decay * e + (1.0 - config.ema_decay) * c,
            ema,
            current,
        )

    return ema_step


def make_update_shard(config: TrainingConfig, static, optimizer):
    @partial(jax.pmap, axis_name="devices")
    def update_shard(params, opt_state, batch, indices, rng, entropy_coefficient):
        # Split on-device (was an eager vmap over device keys per epoch);
        # keys[0]/keys[1] match the previous split_keys[:, 0]/[:, 1] exactly.
        keys = jax.random.split(rng)
        next_rng, update_rng = keys[0], keys[1]
        shard_network = eqx.combine(params, static)
        shard_network, opt_state, metrics = ppo_epoch(
            shard_network,
            opt_state,
            batch,
            indices,
            optimizer,
            update_rng,
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
        return params, opt_state, next_rng, metrics

    return update_shard


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
    # EMA lives replicated on-device so its update is one compiled dispatch
    # per iteration instead of eager per-leaf ops. Checkpoints still store the
    # unreplicated tree (shard 0 is sliced at save; replicated again on load).
    ema_parameters, _ = eqx.partition(ema_network, eqx.is_inexact_array)
    ema_parameters = _replicate_for_pmap(ema_parameters)

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

    kept_samples = int(
        2 * config.num_envs * config.num_steps * config.advantage_top_fraction
    )
    prepare_batch = make_prepare_batch(config)
    ema_step = make_ema_step(config)
    update_shard = make_update_shard(config, static, optimizer)

    device_count = jax.device_count()
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
        if config.debug_timing:
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

        advantages, returns, sample_indices, prep_metrics = prepare_batch(
            rewards, values, next_values, terminated, truncated, winners
        )
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
        entropy_by_device = np.full(
            (device_count,), entropy_coefficient, dtype=np.float32
        )
        epochs_used = 0
        metrics = None
        for _ in range(config.ppo_epochs):
            parameters, optimizer_state, rollout_keys, metrics = update_shard(
                parameters,
                optimizer_state,
                batch,
                sample_indices,
                rollout_keys,
                entropy_by_device,
            )
            epochs_used += 1
            # The KL early stop only matters when there is a next epoch to
            # skip; reading it at ppo_epochs == 1 would just force a device
            # sync for nothing.
            if epochs_used < config.ppo_epochs:
                if float(metrics["approximate_kl"][0]) > config.target_kl:
                    break
        if config.debug_timing:
            jax.block_until_ready(parameters)
        update_seconds = time.perf_counter() - update_started

        ema_parameters = ema_step(ema_parameters, parameters)

        # One transfer for everything the host needs this iteration: the
        # tiny replicated metric arrays come down together and shard 0 is
        # picked on the host, replacing ~20 individual blocking pulls.
        host_metrics = jax.device_get({**prep_metrics, **metrics})
        host_metrics = {name: value[0] for name, value in host_metrics.items()}
        episodes = int(host_metrics.pop("episodes"))
        wins = int(host_metrics.pop("wins"))
        losses = int(host_metrics.pop("losses"))
        draws = episodes - wins - losses
        explained_variance = float(host_metrics.pop("explained_variance"))

        iteration_seconds = time.perf_counter() - iteration_started
        total_samples = device_count * 2 * config.num_envs * config.num_steps
        samples_per_second = total_samples / iteration_seconds
        current_learning_rate = _learning_rate_float(
            config,
            iteration * (config.ppo_epochs * kept_samples // config.minibatch_size),
        )
        record = {
            "iteration": iteration + 1,
            "wall_seconds": time.perf_counter() - train_started,
            "stage": stage_index,
            "samples_per_second": samples_per_second,
            "episodes": episodes,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "explained_variance": explained_variance,
            "learning_rate": current_learning_rate,
            "entropy_coefficient": entropy_coefficient,
            "epochs_used": epochs_used,
            **{name: float(value) for name, value in host_metrics.items()},
        }
        if config.debug_timing:
            record["rollout_seconds"] = rollout_seconds
            record["update_seconds"] = update_seconds

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
            ema_network = eqx.combine(
                jax.tree.map(lambda value: value[0], ema_parameters), static
            )
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
            current_network = eqx.combine(
                jax.tree.map(lambda value: value[0], parameters), static
            )
            current_optimizer_state = jax.tree.map(
                lambda value: value[0], optimizer_state
            )
            ema_network = eqx.combine(
                jax.tree.map(lambda value: value[0], ema_parameters), static
            )
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
    final_ema_network = eqx.combine(
        jax.tree.map(lambda value: value[0], ema_parameters), static
    )
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
    # Persistent XLA compilation cache: first run on a machine pays compile
    # once, every later process starts in seconds. Respect an explicit
    # JAX_COMPILATION_CACHE_DIR; otherwise default to a per-user cache dir.
    if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        jax.config.update(
            "jax_compilation_cache_dir",
            str(Path.home() / ".cache" / "jax_compilation"),
        )
    args = parse_args()
    config = TrainingConfig.from_toml(args.config)
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()

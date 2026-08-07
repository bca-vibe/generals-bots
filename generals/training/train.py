"""Command-line entry point for competition baseline pre-training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import time
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from generals.core.env import GeneralsEnv
from generals.core.game import get_observation

from .actions import CELL_COUNT, MOVE_PLANES, PASS_INDEX
from .config import CurriculumStage, TrainingConfig
from .counterfactual import (
    CastleCounterfactualBuffer,
    generate_counterfactual_refresh,
)
from .conv_model import ConvCompetitionTransformer, calibrate_conv_token_rms
from .evaluation import (
    evaluate_paired_networks,
    evaluate_paired_vs_opponent,
    evaluate_paired_vs_random,
)
from .league import aggregate_league_results, make_opponent_policy
from .model import CompetitionTransformer
from .observation import augment_observation, init_observation_memory
from .ppo import compute_gae, counterfactual_ppo_epoch, ppo_epoch
from .rollout import collect_self_play_rollout
from .tracking import WandbTracker


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


def build_network(
    config: TrainingConfig, key
) -> CompetitionTransformer | ConvCompetitionTransformer:
    common = dict(
        board_size=config.pad_to,
        input_channels=config.input_channels,
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
        observation_schema=config.observation_schema,
        key=key,
    )
    if config.model_architecture == "transformer":
        return CompetitionTransformer(**common)
    if config.model_architecture == "conv_transformer":
        return ConvCompetitionTransformer(
            **common,
            conv_channels=config.conv_channels,
            conv_groups=config.conv_groups,
            with_build_difference_head=config.counterfactual_castle_training,
            with_build_kind_head=config.residual_build_kind_head,
        )
    raise ValueError(f"Unsupported model_architecture {config.model_architecture!r}")


def _batched_memory(config: TrainingConfig):
    memory = init_observation_memory(
        config.pad_to, config.history_size, config.temporal_window
    )
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (config.num_envs, *value.shape)), memory
    )


def _conv_calibration_observations(config: TrainingConfig, pool):
    """Build a deterministic two-seat batch from fresh competition states."""
    requested = config.conv_calibration_samples
    state_count = min((requested + 1) // 2, pool.armies.shape[0])
    states = jax.tree.map(lambda value: value[:state_count], pool)
    observations_zero = jax.vmap(lambda state: get_observation(state, 0))(states)
    observations_one = jax.vmap(lambda state: get_observation(state, 1))(states)
    observations = jax.tree.map(
        lambda zero, one: jnp.concatenate([zero, one]),
        observations_zero,
        observations_one,
    )
    board_masks = jnp.concatenate([states.board_mask, states.board_mask])
    sample_count = min(requested, 2 * state_count)
    observations = jax.tree.map(lambda value: value[:sample_count], observations)
    board_masks = board_masks[:sample_count]

    memory = init_observation_memory(
        config.pad_to, config.history_size, config.temporal_window
    )
    memory = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (sample_count, *value.shape)), memory
    )
    augmented, _ = jax.vmap(
        lambda observation, current_memory, board_mask: augment_observation(
            observation,
            current_memory,
            board_mask,
            config.observation_schema,
            config.deathtouch_turn,
        )
    )(observations, memory, board_masks)
    return augmented


def _replicate_for_pmap(tree):
    """Stage replicas on the host, then place one leading-axis shard per device.

    ``jnp.broadcast_to`` stages the full stacked value on JAX's default device
    before pmap reshards it.  Large environment pools can therefore exhaust the
    first GPU while the other devices still have substantial headroom.
    """
    devices = np.asarray(jax.local_devices())
    device_count = len(devices)
    sharding = NamedSharding(Mesh(devices, ("pmap_devices",)), P("pmap_devices"))

    def replicate(value):
        host_value = np.asarray(jax.device_get(value))
        host_replicas = np.stack([host_value] * device_count)
        return jax.device_put(host_replicas, sharding)

    return jax.tree.map(replicate, tree)


def _reset_replicated_pool(environment: GeneralsEnv, key):
    """Generate the same pool directly on every pmap device.

    Building the full pool eagerly first materializes it on GPU 0.  Even after
    replication and deletion, XLA's BFC allocator retains that extra chunk,
    leaving GPU 0 without enough physical headroom for cuBLAS and the first PPO
    update.  Resetting under pmap places exactly one identical pool replica on
    each device and never creates the additional GPU-0 copy.
    """
    devices = np.asarray(jax.local_devices())
    device_count = len(devices)
    sharding = NamedSharding(Mesh(devices, ("pmap_devices",)), P("pmap_devices"))
    host_key = np.asarray(jax.device_get(key))
    host_keys = np.stack([host_key] * device_count)
    device_keys = jax.device_put(host_keys, sharding)
    return jax.pmap(lambda shard_key: environment.reset(shard_key)[0])(device_keys)


def _first_pmap_replica_slice(tree, count: int):
    """Return a small device-resident slice from replica zero."""

    def first_slice(value):
        shard = value.addressable_shards[0].data
        if shard.ndim and shard.shape[0] == 1:
            shard = shard[0]
        return shard[:count]

    return jax.tree.map(first_slice, tree)


def _first_pmap_replica_to_host(tree):
    """Copy replica zero to host without gathering the global sharded array."""

    def first(value):
        if not isinstance(value, jax.Array):
            return value
        shard = value.addressable_shards[0].data
        if shard.ndim and shard.shape[0] == 1:
            shard = shard[0]
        return np.asarray(jax.device_get(shard))

    return jax.tree.map(
        first,
        tree,
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


def _castle_intervention_strength(
    config: TrainingConfig, iteration_number: int
) -> float:
    """Return the treatment strength for one one-based global iteration."""
    full_until = config.castle_intervention_full_until
    anneal_until = config.castle_intervention_anneal_until
    if full_until <= 0:
        return 0.0
    if iteration_number <= full_until:
        return 1.0
    if anneal_until <= full_until or iteration_number >= anneal_until:
        return 0.0
    return (anneal_until - iteration_number) / (anneal_until - full_until)


def _linear_anneal_multiplier(iteration_number: int, start: int, end: int) -> float:
    """One through ``start``, linear to zero at ``end``; 0/0 means constant one."""
    if start == 0 and end == 0:
        return 1.0
    if iteration_number <= start:
        return 1.0
    if iteration_number >= end:
        return 0.0
    return (end - iteration_number) / (end - start)


def _write_metrics(path: Path, metrics: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_learned_league_manifest(
    run_dir: Path, members: dict[str, list[dict]]
) -> None:
    _write_json_atomic(
        run_dir / "learned_league_manifest.json",
        {
            "schema": "separate_raw_ema_growing_checkpoint_league_members",
            "policies": {
                policy_name: [
                    {
                        "name": member["name"],
                        "iteration": member["iteration"],
                        "sha256": member["sha256"],
                    }
                    for member in policy_members
                ]
                for policy_name, policy_members in members.items()
            },
        },
    )


def _latest_counterfactual_generator_iteration(buffer) -> int | None:
    if buffer is None or buffer.data is None or buffer.size == 0:
        return None
    return int(buffer.data["generator_iteration"].max())


def _request_checkpoint_publication(run_dir: Path, metadata: dict) -> dict:
    """Publish a durable handoff record without waiting for any network upload."""
    record = {
        "event": "requested",
        "iteration": int(metadata["iteration"]),
        "checkpoint_path": metadata["path"],
        "checkpoint_sha256": metadata["sha256"],
        "checkpoint_bytes": int(metadata["bytes"]),
        "raw_weights_present": True,
        "optimizer_state_present": True,
        "ema_weights_present": True,
        "requested": True,
        "complete": False,
        "hash_verified": False,
        "competition_bundle_available": False,
        "raw_competition_bundle_available": False,
    }
    _write_json_atomic(
        run_dir / "publish_requests" / f"checkpoint_{metadata['iteration']:06d}.json",
        record,
    )
    return record


def _ingest_publish_status(
    path: Path, offset: int, tracker: WandbTracker, metrics_path: Path
) -> int:
    """Mirror complete publisher JSONL records into the durable metrics and W&B."""
    if not path.is_file():
        return offset
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        while line := handle.readline():
            if not line.endswith("\n"):
                break
            offset = handle.tell()
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"Warning: ignoring invalid publisher status: {error}")
                continue
            tracker.log_checkpoint_export(record)
            _write_metrics(
                metrics_path,
                {
                    "iteration": record["iteration"],
                    "checkpoint/hf_export_requested": int(
                        record.get("requested", False)
                    ),
                    "checkpoint/hf_export_complete": int(record.get("complete", False)),
                    "checkpoint/hf_hash_verified": int(
                        record.get("hash_verified", False)
                    ),
                    "checkpoint/competition_bundle_available": int(
                        record.get("competition_bundle_available", False)
                    ),
                    "checkpoint/raw_competition_bundle_available": int(
                        record.get("raw_competition_bundle_available", False)
                    ),
                    "checkpoint/hf_upload_seconds": record.get("upload_seconds", 0.0),
                    "checkpoint/hf_remote_path": record.get("remote_path", ""),
                },
            )
    return offset


def _tree_sha256(tree) -> str:
    """Hash array leaves with dtype and shape so matched trunks are auditable."""
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array)):
        array = np.asarray(jax.device_get(leaf))
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _save_checkpoint(
    path: Path,
    network,
    optimizer_state,
    ema_network,
    iteration: int,
    stage_index: int,
    key,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    eqx.tree_serialise_leaves(
        temporary,
        (
            network,
            optimizer_state,
            ema_network,
            jnp.int32(iteration),
            jnp.int32(stage_index),
            key,
        ),
    )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"Checkpoint serialization produced no data: {temporary}")
    os.replace(temporary, path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _save_periodic_checkpoint(
    run_dir: Path,
    parameters,
    static,
    optimizer_state,
    ema_parameters,
    iteration: int,
    stage_index: int,
    key,
    *,
    archive: bool,
) -> dict:
    """Persist a host checkpoint before optional work such as league evaluation."""
    current_network = eqx.combine(_first_pmap_replica_to_host(parameters), static)
    current_optimizer_state = _first_pmap_replica_to_host(optimizer_state)
    ema_network = eqx.combine(_first_pmap_replica_to_host(ema_parameters), static)
    destination = (
        run_dir / f"checkpoint_{iteration:06d}.eqx"
        if archive
        else run_dir / "latest.eqx"
    )
    metadata = _save_checkpoint(
        destination,
        current_network,
        current_optimizer_state,
        ema_network,
        iteration,
        stage_index,
        key,
    )
    if archive:
        _copy_atomic(destination, run_dir / "latest.eqx")
    metadata.update(
        {
            "iteration": iteration,
            "stage": stage_index,
            "archive": archive,
        }
    )
    (run_dir / "latest_checkpoint.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def _load_checkpoint_state(path, skeleton, config: TrainingConfig):
    """Load a checkpoint with an actionable error for schema/shape mismatches."""
    try:
        return eqx.tree_deserialise_leaves(path, skeleton)
    except (EOFError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            f"Could not load checkpoint {path!s} using observation_schema="
            f"{config.observation_schema!r} ({config.input_channels} channels) and "
            f"model_architecture={config.model_architecture!r}. "
            "Use the checkpoint's original run config; legacy checkpoints require "
            "legacy_38 with the transformer architecture."
        ) from error


def _attach_zero_build_difference_head(legacy, initialized):
    """Copy a historical conv actor/critic into a castle-head skeleton."""
    if not isinstance(legacy, ConvCompetitionTransformer):
        raise TypeError("Castle counterfactual migration requires a conv checkpoint")
    if (
        initialized.build_difference_head is None
        and initialized.build_kind_head is None
    ):
        raise ValueError("Migration skeleton has no new castle head")
    return eqx.tree_at(
        lambda network: (network.transformer, network.conv_patch_residual),
        initialized,
        (legacy.transformer, legacy.conv_patch_residual),
    )


def _migrate_optimizer_state_with_zero_head(old_state, new_state):
    """Preserve Adam moments/counts while adding zero moments for the new head."""
    def unpack(state):
        # Depending on the Optax version, the transforms composing adam() are
        # either flattened into the outer chain or retained as a nested pair.
        if len(state) == 3 and hasattr(state[1], "mu"):
            return state[1], state[2], False
        if (
            len(state) == 2
            and isinstance(state[1], tuple)
            and len(state[1]) == 2
            and hasattr(state[1][0], "mu")
        ):
            return state[1][0], state[1][1], True
        raise ValueError(
            "Unexpected optimizer chain while migrating castle head: "
            f"{jax.tree.structure(state)}"
        )

    old_adam, old_schedule, old_nested = unpack(old_state)
    new_adam, new_schedule, new_nested = unpack(new_state)
    if old_nested != new_nested:
        raise ValueError("Optimizer-state layouts differ during castle-head migration")
    migrated_mu = _attach_zero_build_difference_head(old_adam.mu, new_adam.mu)
    migrated_nu = _attach_zero_build_difference_head(old_adam.nu, new_adam.nu)
    migrated_adam = new_adam._replace(
        count=old_adam.count,
        mu=migrated_mu,
        nu=migrated_nu,
    )
    migrated_schedule = new_schedule._replace(count=old_schedule.count)
    if new_nested:
        return (new_state[0], (migrated_adam, migrated_schedule))
    return (new_state[0], migrated_adam, migrated_schedule)


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
            observation_schema=config.observation_schema,
        )
    )


def _make_opponent_evaluator(
    config: TrainingConfig,
    environment,
    n_maps: int,
    truncation: int,
    opponent_name: str,
):
    opponent_action = make_opponent_policy(opponent_name)
    return eqx.filter_jit(
        lambda pool, network, key: evaluate_paired_vs_opponent(
            environment,
            pool,
            network,
            key,
            n_maps,
            truncation,
            opponent_action,
            pad_to=config.pad_to,
            history_size=config.history_size,
            temporal_window=config.temporal_window,
            observation_schema=config.observation_schema,
        )
    )


def _shard_league_pool(pool, n_maps: int, device_count: int):
    if n_maps % device_count:
        raise ValueError(
            f"league_eval_maps ({n_maps}) must be divisible by devices ({device_count})"
        )
    maps_per_device = n_maps // device_count
    return jax.tree.map(
        lambda value: value[:n_maps].reshape(
            (device_count, maps_per_device, *value.shape[1:])
        ),
        pool,
    )


def _shard_replicated_league_pool(pool, n_maps: int, device_count: int):
    """Take disjoint map slices from a pool already replicated by ``pmap``."""
    if n_maps % device_count:
        raise ValueError(
            f"league maps ({n_maps}) must be divisible by devices ({device_count})"
        )
    maps_per_device = n_maps // device_count
    offsets = jnp.arange(device_count, dtype=jnp.int32) * maps_per_device

    def take_slice(pool_shard, offset):
        return jax.tree.map(
            lambda value: jax.lax.dynamic_slice_in_dim(
                value, offset, maps_per_device, axis=0
            ),
            pool_shard,
        )

    return jax.pmap(take_slice)(pool, offsets)


def _safe_rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _behavior_rates(
    totals: dict[str, float], source_prefix: str, output_prefix: str = "behavior/"
) -> dict[str, float]:
    """Turn additive behavior counters into stable, auditable rates."""

    def get(name: str) -> float:
        return totals.get(f"{source_prefix}{name}", 0.0)

    actions = get("actions")
    moves = get("moves")
    builds = get("builds")
    opportunities = get("build_opportunity_steps")
    completed_games = get("completed_games")
    games_with_opportunity = get("games_with_build_opportunity")
    moves_after_move = get("moves_after_move")
    counters = {
        name[len(source_prefix) :]: value
        for name, value in totals.items()
        if name.startswith(source_prefix)
    }
    return {
        **{f"{output_prefix}count/{name}": value for name, value in counters.items()},
        f"{output_prefix}castle_build/action_share": _safe_rate(builds, actions),
        f"{output_prefix}castle_build/legal_step_rate": _safe_rate(
            builds, opportunities
        ),
        f"{output_prefix}castle_build/legal_game_rate": _safe_rate(
            get("games_with_build"), games_with_opportunity
        ),
        f"{output_prefix}castle_build/builds_per_game": _safe_rate(
            builds, completed_games
        ),
        f"{output_prefix}dither/move_rate": _safe_rate(get("dithers"), moves),
        f"{output_prefix}dither/consecutive_move_rate": _safe_rate(
            get("dithers"), moves_after_move
        ),
        f"{output_prefix}pass/rate": _safe_rate(get("passes"), actions),
        f"{output_prefix}half_move/rate": _safe_rate(get("half_moves"), moves),
        f"{output_prefix}movement/reinforcement_rate": _safe_rate(
            get("reinforce_moves"), moves
        ),
        f"{output_prefix}movement/visible_neutral_expansion_rate": _safe_rate(
            get("expansion_moves"), moves
        ),
        f"{output_prefix}movement/visible_opponent_attack_rate": _safe_rate(
            get("attack_moves"), moves
        ),
        f"{output_prefix}game/mean_length": _safe_rate(
            get("game_length_sum"), completed_games
        ),
        f"{output_prefix}game/mean_terminal_land_margin": _safe_rate(
            get("terminal_land_margin_sum"), completed_games
        ),
        f"{output_prefix}game/mean_terminal_army_margin": _safe_rate(
            get("terminal_army_margin_sum"), completed_games
        ),
    }


def _combine_sharded_evaluation(metrics) -> dict[str, float]:
    host = {name: np.asarray(value, dtype=np.float64) for name, value in metrics.items()}
    wins = float(host["wins"].sum())
    losses = float(host["losses"].sum())
    draws = float(host["draws"].sum())
    score = float(host["score"].mean())
    second_moment = np.mean(host["paired_score_std"] ** 2 + host["score"] ** 2)
    paired_score_std = float(np.sqrt(max(0.0, second_moment - score**2)))
    result = {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": wins + losses + draws,
        "score": score,
        "paired_score_std": paired_score_std,
    }
    additive = {
        name: float(value.sum())
        for name, value in host.items()
        if name not in {"wins", "losses", "draws", "score", "paired_score_std"}
    }
    result.update(additive)
    if any(name.startswith("behavior_") for name in additive):
        result.update(_behavior_rates(additive, "behavior_"))
    if any(name.startswith("behavior_a_") for name in additive):
        result.update(_behavior_rates(additive, "behavior_a_", "behavior/current/"))
    if any(name.startswith("behavior_b_") for name in additive):
        result.update(_behavior_rates(additive, "behavior_b_", "behavior/frozen/"))
    return result


def _run_league(
    config: TrainingConfig,
    policies: dict[str, object],
    tracker: WandbTracker,
    run_dir: Path,
    iteration: int,
    *,
    label: str,
    checkpoint_opponent=None,
) -> dict:
    """Evaluate named policies on identical locked final-stage maps, sharded by GPU."""
    stage = config.curriculum[-1]
    largest_map_count = max(config.league_eval_maps, config.league_checkpoint_maps)
    environment = make_environment(config, stage, pool_size=max(largest_map_count, 16))
    pool_key = jax.random.PRNGKey(config.league_eval_seed)
    pool_replicated = _reset_replicated_pool(environment, pool_key)
    device_count = jax.device_count()
    sharded_pool = _shard_replicated_league_pool(
        pool_replicated, config.league_eval_maps, device_count
    )
    maps_per_device = config.league_eval_maps // device_count
    payload_policies: dict[str, dict] = {}
    league_started = time.perf_counter()

    for policy_index, (policy_name, network) in enumerate(policies.items()):
        host_network = jax.device_get(network)
        results: dict[str, dict[str, float]] = {}
        for opponent_index, opponent_name in enumerate(config.league_opponents):
            opponent_action = make_opponent_policy(opponent_name)

            def evaluate_shard(pool_shard, network_shard, evaluation_key):
                return evaluate_paired_vs_opponent(
                    environment,
                    pool_shard,
                    network_shard,
                    evaluation_key,
                    maps_per_device,
                    config.truncation,
                    opponent_action,
                    pad_to=config.pad_to,
                    history_size=config.history_size,
                    temporal_window=config.temporal_window,
                    observation_schema=config.observation_schema,
                )

            evaluator = jax.pmap(evaluate_shard, in_axes=(0, None, 0))
            evaluation_key = jax.random.fold_in(
                pool_key, policy_index * len(config.league_opponents) + opponent_index + 1
            )
            device_keys = jax.random.split(evaluation_key, device_count)
            opponent_started = time.perf_counter()
            evaluation, _ = evaluator(sharded_pool, host_network, device_keys)
            evaluation = jax.device_get(evaluation)
            result = _combine_sharded_evaluation(evaluation)
            result["evaluation_seconds"] = time.perf_counter() - opponent_started
            results[opponent_name] = result
            record = {
                "iteration": iteration,
                **{
                    f"league/{policy_name}/{opponent_name}/{name}": value
                    for name, value in result.items()
                },
            }
            _write_metrics(run_dir / "metrics.jsonl", record)
            tracker.log_evaluation(record)
            print(
                f"  league {policy_name}/{opponent_name}: {int(result['wins'])}W/"
                f"{int(result['losses'])}L/{int(result['draws'])}D, "
                f"score={result['score']:.3f}"
            )

        aggregate = None
        if results:
            aggregate = aggregate_league_results(results)
            behavior_totals: dict[str, float] = {}
            for result in results.values():
                for name, value in result.items():
                    if name.startswith("behavior_"):
                        behavior_totals[name] = behavior_totals.get(name, 0.0) + value
            aggregate.update(behavior_totals)
            aggregate.update(_behavior_rates(behavior_totals, "behavior_"))
            aggregate["evaluation_seconds"] = sum(
                result["evaluation_seconds"] for result in results.values()
            )
            aggregate_record = {
                "iteration": iteration,
                **{
                    f"league/{policy_name}/aggregate/{name}": value
                    for name, value in aggregate.items()
                },
            }
            _write_metrics(run_dir / "metrics.jsonl", aggregate_record)
            tracker.log_evaluation(aggregate_record)
        checkpoint_result = None
        seven_opponent_macro_score = None
        if checkpoint_opponent is not None:
            checkpoint_maps = config.league_checkpoint_maps
            if checkpoint_maps % device_count:
                raise ValueError(
                    "league_checkpoint_maps must be divisible by the device count"
                )
            checkpoint_pool = _shard_replicated_league_pool(
                pool_replicated, checkpoint_maps, device_count
            )
            checkpoint_maps_per_device = checkpoint_maps // device_count

            def evaluate_checkpoint_shard(pool_shard, current, frozen):
                return evaluate_paired_networks(
                    environment,
                    pool_shard,
                    current,
                    frozen,
                    checkpoint_maps_per_device,
                    config.truncation,
                    schema_a=config.observation_schema,
                    schema_b=config.observation_schema,
                    pad_to=config.pad_to,
                    history_size=config.history_size,
                    temporal_window=config.temporal_window,
                )

            checkpoint_evaluator = jax.pmap(
                evaluate_checkpoint_shard, in_axes=(0, None, None)
            )
            checkpoint_started = time.perf_counter()
            checkpoint_metrics = checkpoint_evaluator(
                checkpoint_pool, host_network, jax.device_get(checkpoint_opponent)
            )
            checkpoint_result = _combine_sharded_evaluation(
                jax.device_get(checkpoint_metrics)
            )
            checkpoint_result["evaluation_seconds"] = (
                time.perf_counter() - checkpoint_started
            )
            checkpoint_name = config.league_checkpoint_name or "fixed_checkpoint"
            checkpoint_record = {
                "iteration": iteration,
                **{
                    f"league/{policy_name}/{checkpoint_name}/{name}": value
                    for name, value in checkpoint_result.items()
                },
            }
            seven_scores = [result["score"] for result in results.values()]
            seven_scores.append(checkpoint_result["score"])
            seven_opponent_macro_score = float(np.mean(seven_scores))
            checkpoint_record[
                f"league/{policy_name}/seven_opponent_macro_score"
            ] = seven_opponent_macro_score
            _write_metrics(run_dir / "metrics.jsonl", checkpoint_record)
            tracker.log_evaluation(checkpoint_record)
            print(
                f"  league {policy_name}/{checkpoint_name}: "
                f"{int(checkpoint_result['wins'])}W/"
                f"{int(checkpoint_result['losses'])}L/"
                f"{int(checkpoint_result['draws'])}D, "
                f"score={checkpoint_result['score']:.3f}"
            )
        payload_policies[policy_name] = {
            "opponents": results,
            "aggregate": aggregate,
            "fixed_checkpoint": checkpoint_result,
            "seven_opponent_macro_score": seven_opponent_macro_score,
        }
        if aggregate is not None:
            print(
                f"  league {policy_name}/aggregate: {int(aggregate['wins'])}W/"
                f"{int(aggregate['losses'])}L/{int(aggregate['draws'])}D, "
                f"score={aggregate['score']:.3f}, macro={aggregate['macro_score']:.3f}"
            )

    payload = {
        "iteration": iteration,
        "label": label,
        "seed": config.league_eval_seed,
        "maps_per_opponent": config.league_eval_maps,
        "games_per_opponent": 2 * config.league_eval_maps,
        "policies": payload_policies,
        "evaluation_seconds": time.perf_counter() - league_started,
    }
    result_path = run_dir / f"league_{label}.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def _evaluate_network_pair(
    config: TrainingConfig,
    network_a,
    network_b,
    *,
    n_maps: int,
    seed: int,
) -> dict[str, float]:
    """Greedy paired-seat evaluation of two learned policies on locked maps."""
    device_count = jax.device_count()
    if n_maps % device_count:
        raise ValueError(
            f"learned league maps ({n_maps}) must be divisible by {device_count} devices"
        )
    stage = config.curriculum[-1]
    environment = make_environment(config, stage, pool_size=max(n_maps, 16))
    pool_replicated = _reset_replicated_pool(environment, jax.random.PRNGKey(seed))
    sharded_pool = _shard_replicated_league_pool(
        pool_replicated, n_maps, device_count
    )
    maps_per_device = n_maps // device_count

    def evaluate_shard(pool_shard, current, frozen):
        return evaluate_paired_networks(
            environment,
            pool_shard,
            current,
            frozen,
            maps_per_device,
            config.truncation,
            schema_a=config.observation_schema,
            schema_b=config.observation_schema,
            pad_to=config.pad_to,
            history_size=config.history_size,
            temporal_window=config.temporal_window,
        )

    evaluator = jax.pmap(evaluate_shard, in_axes=(0, None, None))
    metrics = evaluator(
        sharded_pool,
        jax.device_get(network_a),
        jax.device_get(network_b),
    )
    return _combine_sharded_evaluation(jax.device_get(metrics))


def _run_learned_checkpoint_league(
    config: TrainingConfig,
    current_policies: dict[str, object],
    members: dict[str, list[dict]],
    tracker: WandbTracker,
    run_dir: Path,
    iteration: int,
) -> dict:
    """Evaluate raw and EMA only against the corresponding frozen snapshots."""
    started = time.perf_counter()
    device_count = jax.device_count()
    if config.learned_league_maps % device_count:
        raise ValueError(
            "learned_league_maps must be divisible by the JAX device count"
        )
    stage = config.curriculum[-1]
    environment = make_environment(
        config, stage, pool_size=max(config.learned_league_maps, 16)
    )
    pool_replicated = _reset_replicated_pool(
        environment, jax.random.PRNGKey(config.learned_league_seed)
    )
    sharded_pool = _shard_replicated_league_pool(
        pool_replicated, config.learned_league_maps, device_count
    )
    maps_per_device = config.learned_league_maps // device_count

    def evaluate_shard(pool_shard, current, frozen):
        return evaluate_paired_networks(
            environment,
            pool_shard,
            current,
            frozen,
            maps_per_device,
            config.truncation,
            schema_a=config.observation_schema,
            schema_b=config.observation_schema,
            pad_to=config.pad_to,
            history_size=config.history_size,
            temporal_window=config.temporal_window,
        )

    evaluator = jax.pmap(evaluate_shard, in_axes=(0, None, None))
    payload: dict[str, object] = {
        "schema": "separate_raw_ema_growing_checkpoint_league",
        "iteration": iteration,
        "seed": config.learned_league_seed,
        "maps_per_matchup": config.learned_league_maps,
        "games_per_matchup": 2 * config.learned_league_maps,
        "policies": {},
    }
    for policy_index, policy_name in enumerate(("raw", "ema")):
        current = current_policies[policy_name]
        policy_results: dict[str, dict[str, float]] = {}
        for opponent_index, member in enumerate(members[policy_name]):
            matchup_started = time.perf_counter()
            # Identical boards for every matchup and both policy families.
            metrics = evaluator(
                sharded_pool,
                jax.device_get(current),
                jax.device_get(member["network"]),
            )
            result = _combine_sharded_evaluation(jax.device_get(metrics))
            result["evaluation_seconds"] = time.perf_counter() - matchup_started
            policy_results[member["name"]] = result
            record = {
                "iteration": iteration,
                **{
                    f"learned_league/{policy_name}/{member['name']}/{name}": value
                    for name, value in result.items()
                },
            }
            _write_metrics(run_dir / "metrics.jsonl", record)
            tracker.log_evaluation(record)
            print(
                f"  learned league {policy_name}/{member['name']}: "
                f"{int(result['wins'])}W/{int(result['losses'])}L/"
                f"{int(result['draws'])}D, score={result['score']:.3f}"
            )
        macro_score = float(
            np.mean([result["score"] for result in policy_results.values()])
        )
        aggregate_record = {
            "iteration": iteration,
            f"learned_league/{policy_name}/aggregate/macro_score": macro_score,
            f"learned_league/{policy_name}/aggregate/opponents": len(policy_results),
        }
        _write_metrics(run_dir / "metrics.jsonl", aggregate_record)
        tracker.log_evaluation(aggregate_record)
        payload["policies"][policy_name] = {
            "opponents": policy_results,
            "macro_score": macro_score,
            "member_iterations": [member["iteration"] for member in members[policy_name]],
        }
    payload["evaluation_seconds"] = time.perf_counter() - started
    _write_json_atomic(
        run_dir / f"learned_league_{iteration:06d}.json", payload
    )
    return payload


def _run_post_training_league(
    config: TrainingConfig,
    network,
    tracker: WandbTracker,
    run_dir: Path,
    *,
    iteration: int | None = None,
) -> dict:
    """Backward-compatible EMA-only terminal league entrypoint."""
    return _run_league(
        config,
        {"ema": network},
        tracker,
        run_dir,
        config.num_iterations if iteration is None else iteration,
        label="final",
    )


def make_prepare_batch(config: TrainingConfig):
    """GAE -> returns -> normalization -> top-k selection -> prep metrics,
    fused into one compiled dispatch (was five dispatches plus eager glue)."""
    kept_samples = int(
        2 * config.num_envs * config.num_steps * config.advantage_top_fraction
    )

    @partial(jax.pmap, axis_name="devices")
    def prepare_batch(
        rewards,
        actor_rewards,
        potential_rewards,
        values,
        next_values,
        terminated,
        truncated,
        winners,
        actions,
        tactical_build_masks,
        selected_tactical_build,
        base_policy_statistics,
        behavior_policy_statistics,
    ):
        raw_advantages = compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            config.gamma,
            config.gae_lambda,
        )
        actor_advantages = compute_gae(
            actor_rewards,
            values,
            next_values,
            terminated,
            truncated,
            config.gamma,
            config.gae_lambda,
        )
        returns = raw_advantages + values

        actor_mean = jax.lax.pmean(actor_advantages.mean(), axis_name="devices")
        actor_mean_square = jax.lax.pmean(
            (actor_advantages**2).mean(), axis_name="devices"
        )
        actor_std = jnp.sqrt(
            jnp.maximum(actor_mean_square - actor_mean**2, 0.0)
        )
        raw_mean = jax.lax.pmean(raw_advantages.mean(), axis_name="devices")
        raw_mean_square = jax.lax.pmean(
            (raw_advantages**2).mean(), axis_name="devices"
        )
        raw_std = jnp.sqrt(jnp.maximum(raw_mean_square - raw_mean**2, 0.0))
        normalized = (actor_advantages - actor_mean) / (actor_std + 1e-8)
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

        build_start = MOVE_PLANES * CELL_COUNT
        build_actions = (actions >= build_start) & (actions < PASS_INDEX)
        build_count = jax.lax.psum(build_actions.sum(), axis_name="devices")
        retained_build_count = jax.lax.psum(
            build_actions.reshape(-1)[indices].sum(), axis_name="devices"
        )
        eligible_steps = jax.lax.psum(
            jnp.any(tactical_build_masks, axis=-1).sum(), axis_name="devices"
        )
        selected_tactical_builds = jax.lax.psum(
            selected_tactical_build.sum(), axis_name="devices"
        )

        def masked_global_mean(values, condition):
            total = jax.lax.psum(
                jnp.where(condition, values, 0.0).sum(), axis_name="devices"
            )
            count = jax.lax.psum(condition.sum(), axis_name="devices")
            return total / jnp.maximum(count, 1)

        prep_metrics = {
            "raw_advantage_std": raw_std,
            "actor_advantage_std": actor_std,
            "mean_reward": jax.lax.pmean(rewards.mean(), axis_name="devices"),
            "actor_mean_reward": jax.lax.pmean(
                actor_rewards.mean(), axis_name="devices"
            ),
            "potential_reward_mean": jax.lax.pmean(
                potential_rewards.mean(), axis_name="devices"
            ),
            "potential_reward_abs_mean": jax.lax.pmean(
                jnp.abs(potential_rewards).mean(), axis_name="devices"
            ),
            "episodes": episodes,
            "wins": wins,
            "losses": losses,
            "explained_variance": explained_variance,
            "tactical_eligible_steps": eligible_steps,
            "tactical_selected_builds": selected_tactical_builds,
            "rollout_builds": build_count,
            "retained_builds": retained_build_count,
            "build_top_filter_retention": retained_build_count
            / jnp.maximum(build_count, 1),
            "build_actor_advantage_mean": masked_global_mean(
                actor_advantages, build_actions
            ),
            "build_raw_advantage_mean": masked_global_mean(
                raw_advantages, build_actions
            ),
        }
        for name, values in base_policy_statistics.items():
            prep_metrics[f"underlying_policy_{name}"] = jax.lax.pmean(
                values.mean(), axis_name="devices"
            )
        for name, values in behavior_policy_statistics.items():
            prep_metrics[f"behavior_policy_{name}"] = jax.lax.pmean(
                values.mean(), axis_name="devices"
            )
        return normalized, returns, indices, prep_metrics

    return prepare_batch


def make_natural_castle_metrics(config: TrainingConfig):
    """Count only actions actually sampled in the ordinary PPO stream."""

    @partial(jax.pmap, axis_name="devices")
    def natural_metrics(
        build_actions,
        legal_build_opportunities,
        completed_game_with_build,
        completed_player_game_with_build,
        completed_player_games,
        completed_game_builds,
        terminated,
        truncated,
    ):
        psum = lambda value: jax.lax.psum(value, axis_name="devices")
        decisions = psum(jnp.asarray(build_actions.size, dtype=jnp.int32))
        builds = psum(build_actions.sum())
        eligible = psum(legal_build_opportunities.sum())
        completed_games = psum(completed_player_games[:, : config.num_envs].sum())
        completed_player_game_count = psum(completed_player_games.sum())
        games_with_build = psum(completed_game_with_build.sum())
        player_games_with_build = psum(completed_player_game_with_build.sum())
        p0_games_with_build = psum(
            completed_player_game_with_build[:, : config.num_envs].sum()
        )
        p1_games_with_build = psum(
            completed_player_game_with_build[:, config.num_envs :].sum()
        )
        p0_games = psum(completed_player_games[:, : config.num_envs].sum())
        p1_games = psum(completed_player_games[:, config.num_envs :].sum())
        return {
            "ppo_castle/count/builds": builds,
            "ppo_castle/count/completed_game_builds": psum(
                completed_game_builds.sum()
            ),
            "ppo_castle/count/decisions": decisions,
            "ppo_castle/count/eligible_decisions": eligible,
            "ppo_castle/count/completed_games": completed_games,
            "ppo_castle/count/completed_player_games": completed_player_game_count,
            "ppo_castle/count/games_with_build": games_with_build,
            "ppo_castle/count/player_games_with_build": player_games_with_build,
            "ppo_castle/count/terminated_games": psum(
                terminated[:, : config.num_envs].sum()
            ),
            "ppo_castle/count/truncated_games": psum(
                truncated[:, : config.num_envs].sum()
            ),
            "ppo_castle/count/seat0_games_with_build": p0_games_with_build,
            "ppo_castle/count/seat1_games_with_build": p1_games_with_build,
            "ppo_castle/count/seat0_completed_games": p0_games,
            "ppo_castle/count/seat1_completed_games": p1_games,
        }

    return natural_metrics


def _natural_castle_rates(counts: dict[str, float], prefix="ppo_castle/"):
    get = lambda name: counts.get(f"ppo_castle/count/{name}", 0.0)
    safe = lambda numerator, denominator: numerator / denominator if denominator else 0.0
    return {
        f"{prefix}build_move_rate": safe(get("builds"), get("decisions")),
        f"{prefix}build_game_rate": safe(
            get("games_with_build"), get("completed_games")
        ),
        f"{prefix}build_player_game_rate": safe(
            get("player_games_with_build"), get("completed_player_games")
        ),
        f"{prefix}builds_per_game": safe(
            get("completed_game_builds"), get("completed_games")
        ),
        f"{prefix}build_eligible_rate": safe(
            get("builds"), get("eligible_decisions")
        ),
        f"{prefix}seat0_build_game_rate": safe(
            get("seat0_games_with_build"), get("seat0_completed_games")
        ),
        f"{prefix}seat1_build_game_rate": safe(
            get("seat1_games_with_build"), get("seat1_completed_games")
        ),
    }


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
    def update_shard(
        params,
        opt_state,
        batch,
        indices,
        rng,
        entropy_coefficient,
        tactical_build_logit_boost,
    ):
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
            tactical_build_logit_boost=tactical_build_logit_boost,
            axis_name="devices",
        )
        params, _ = eqx.partition(shard_network, eqx.is_inexact_array)
        metrics = jax.lax.pmean(metrics, axis_name="devices")
        return params, opt_state, next_rng, metrics

    return update_shard


def make_counterfactual_update_shard(config: TrainingConfig, static, optimizer):
    """Treatment-only pmap; the disabled control never traces this function."""

    @partial(jax.pmap, axis_name="devices")
    def update_shard(
        params,
        opt_state,
        batch,
        indices,
        counterfactual,
        rng,
        entropy_coefficient,
        actor_coefficient_scale,
        critic_coefficient_scale,
    ):
        keys = jax.random.split(rng)
        next_rng, update_rng = keys[0], keys[1]
        shard_network = eqx.combine(params, static)
        shard_network, opt_state, metrics = counterfactual_ppo_epoch(
            shard_network,
            opt_state,
            batch,
            indices,
            counterfactual,
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
            actor_coefficient=config.counterfactual_actor_coefficient,
            successor_coefficient=config.counterfactual_value_coefficient,
            delta_coefficient=config.counterfactual_delta_coefficient,
            actor_temperature=config.counterfactual_actor_temperature,
            actor_weight_scale=config.counterfactual_actor_weight_scale,
            huber_delta=config.counterfactual_huber_delta,
            actor_coefficient_scale=actor_coefficient_scale,
            critic_coefficient_scale=critic_coefficient_scale,
            axis_name="devices",
        )
        params, _ = eqx.partition(shard_network, eqx.is_inexact_array)
        metrics = jax.lax.pmean(metrics, axis_name="devices")
        return params, opt_state, next_rng, metrics

    return update_shard


def train(
    config: TrainingConfig,
    *,
    resume: str | None = None,
    trace_dir: str | None = None,
    trace_start_iteration: int = 1,
    trace_iterations: int = 0,
    stop_at_unix: float | None = None,
    duration_seconds: float | None = None,
    initialization_gate: str | None = None,
    graceful_signals: bool = False,
):
    """Train the shared AverageJoe-style baseline and return its final state."""
    config.validate()
    if trace_iterations < 0:
        raise ValueError("trace_iterations must be non-negative")
    if trace_iterations and trace_start_iteration <= 0:
        raise ValueError("trace_start_iteration must be positive")
    if trace_iterations and not trace_dir:
        raise ValueError("trace_dir is required when trace_iterations is non-zero")
    if stop_at_unix is not None and duration_seconds is not None:
        raise ValueError("stop_at_unix and duration_seconds are mutually exclusive")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    stop_requested = False
    previous_signal_handlers: dict[int, object] = {}

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"Graceful stop requested by signal {signum}; finishing current operation")

    if graceful_signals:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    metrics_path = run_dir / "metrics.jsonl"

    device_count = jax.device_count()
    print(f"Devices ({device_count}): {jax.devices()}")
    expected_device_count = os.environ.get("EXPECTED_JAX_DEVICE_COUNT")
    if expected_device_count is not None and device_count != int(expected_device_count):
        raise RuntimeError(
            f"Expected {expected_device_count} visible JAX devices, found "
            f"{device_count}: {jax.devices()}"
        )
    print(
        f"Competition baseline: L={config.depth}, d={config.embed_dim}, "
        f"heads={config.attention_heads}, architecture={config.model_architecture}, "
        f"envs/device={config.num_envs}"
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
    fixed_league_network = None
    if resume:
        actual_resume_sha256 = _sha256_file(Path(resume))
        resume_is_parent_checkpoint = (
            config.resume_checkpoint_sha256 is None
            or actual_resume_sha256 == config.resume_checkpoint_sha256
        )
        if config.resume_checkpoint_sha256:
            recovery_inside_run = (
                Path(resume).resolve().parent == run_dir.resolve()
                or run_dir.resolve() in Path(resume).resolve().parents
            )
            if not resume_is_parent_checkpoint and not recovery_inside_run:
                raise ValueError(
                    f"Resume checkpoint SHA-256 mismatch: expected "
                    f"{config.resume_checkpoint_sha256}, got {actual_resume_sha256}"
                )
        if (
            config.counterfactual_castle_training
            or config.residual_build_kind_head
        ) and resume_is_parent_checkpoint and not (
            config.resume_checkpoint_has_counterfactual_heads
        ):
            historical_config = replace(
                config,
                counterfactual_castle_training=False,
                residual_build_kind_head=False,
            )
            historical_network = build_network(historical_config, network_key)
            historical_optimizer_state = optimizer.init(
                eqx.filter(historical_network, eqx.is_inexact_array)
            )
            historical_skeleton = (
                historical_network,
                historical_optimizer_state,
                historical_network,
                jnp.int32(0),
                jnp.int32(0),
                key,
            )
            (
                loaded_raw,
                loaded_optimizer_state,
                loaded_ema,
                saved_iteration,
                saved_stage_index,
                key,
            ) = _load_checkpoint_state(resume, historical_skeleton, historical_config)
            network = _attach_zero_build_difference_head(loaded_raw, network)
            ema_network = _attach_zero_build_difference_head(loaded_ema, ema_network)
            optimizer_state = _migrate_optimizer_state_with_zero_head(
                loaded_optimizer_state,
                optimizer_state,
            )
            _write_json_atomic(
                run_dir / "checkpoint_schema.json",
                {
                    "schema": "castle_counterfactual_training",
                    "version": config.counterfactual_schema_version,
                    "source_checkpoint": str(resume),
                    "source_checkpoint_sha256": actual_resume_sha256,
                    "auxiliary_head": "spatial_build_difference",
                    "auxiliary_head_initialization": "zeros",
                    "actor_build_kind_head": "scalar_residual",
                    "actor_build_kind_head_initialization": "zeros",
                    "actor_build_kind_head_deployment_visible": True,
                },
            )
        else:
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
            ) = _load_checkpoint_state(resume, skeleton, config)
        if (
            config.counterfactual_castle_training
            and config.resume_checkpoint_has_counterfactual_heads
            and not (run_dir / "checkpoint_schema.json").exists()
        ):
            _write_json_atomic(
                run_dir / "checkpoint_schema.json",
                {
                    "schema": "castle_counterfactual_training",
                    "version": config.counterfactual_schema_version,
                    "source_checkpoint": str(resume),
                    "source_checkpoint_sha256": actual_resume_sha256,
                    "auxiliary_head": "spatial_build_difference",
                    "auxiliary_head_initialization": "resumed",
                    "actor_build_kind_head": "scalar_residual",
                    "actor_build_kind_head_initialization": "resumed",
                    "actor_build_kind_head_deployment_visible": True,
                },
            )
        start_iteration = int(saved_iteration)
        stage_index = int(saved_stage_index)
        if stage_index >= len(config.curriculum):
            raise ValueError(
                f"Checkpoint curriculum stage {stage_index} is not present in the config"
            )
        if (
            config.parent_final_iteration
            and resume_is_parent_checkpoint
            and start_iteration != config.parent_final_iteration
        ):
            raise ValueError(
                f"Expected parent iteration {config.parent_final_iteration}, "
                f"checkpoint contains {start_iteration}"
            )
        if config.resume_start_stage >= 0 and stage_index != config.resume_start_stage:
            raise ValueError(
                f"Expected resume stage {config.resume_start_stage}, "
                f"checkpoint contains {stage_index}"
            )
        if config.league_checkpoint_path and (
            Path(config.league_checkpoint_path).resolve() == Path(resume).resolve()
            or config.league_checkpoint_sha256 == actual_resume_sha256
        ):
            fixed_league_network = (
                ema_network if config.league_checkpoint_policy == "ema" else network
            )
        print(
            f"Resumed {resume} at iteration {start_iteration}, curriculum stage {stage_index}"
        )

    trainable = eqx.filter(network, eqx.is_inexact_array)
    parameter_count = sum(value.size for value in jax.tree.leaves(trainable))
    print(f"Array parameters: {parameter_count:,}")
    tracker = WandbTracker.start(
        config,
        start_iteration=start_iteration,
        resume=resume,
        device_count=device_count,
    )

    stage = config.curriculum[stage_index]
    environment = make_environment(config, stage)
    key, pool_key = jax.random.split(key)
    pool_replicated = _reset_replicated_pool(environment, pool_key)
    trunk = network.transformer if isinstance(network, ConvCompetitionTransformer) else network
    initialization = {
        "parameter_count": parameter_count,
        "transformer_trunk_sha256": _tree_sha256(trunk),
        "architecture": config.model_architecture,
        "seed": config.seed,
        "start_iteration": start_iteration,
        "resume_raw_weights": bool(resume and config.resume_raw_weights),
        "resume_optimizer_state": bool(resume and config.resume_optimizer_state),
        "resume_ema_weights": bool(resume and config.resume_ema_weights),
        "parent_wall_seconds": config.parent_wall_seconds,
    }
    if not resume and isinstance(network, ConvCompetitionTransformer):
        calibration_state_count = (config.conv_calibration_samples + 1) // 2
        calibration_pool = _first_pmap_replica_slice(
            pool_replicated, calibration_state_count
        )
        calibration_observations = _conv_calibration_observations(
            config, calibration_pool
        )
        network, calibration = calibrate_conv_token_rms(
            network,
            calibration_observations,
            config.conv_initial_token_rms_ratio,
        )
        jax.block_until_ready(calibration)
        calibration = {name: float(value) for name, value in calibration.items()}
        calibration["samples"] = int(calibration_observations.shape[0])
        initialization["conv_calibration"] = calibration
        (run_dir / "conv_calibration.json").write_text(
            json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            "Calibrated convolutional token correction: "
            f"ratio {calibration['ratio_before']:.4f} -> "
            f"{calibration['ratio_after']:.4f} "
            f"on {calibration['samples']} observations "
            f"(projection x{calibration['projection_multiplier']:.4f})"
        )
        ema_network = network
        optimizer_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    if config.league_checkpoint_path and fixed_league_network is None:
        fixed_path = Path(config.league_checkpoint_path)
        actual_sha256 = _sha256_file(fixed_path)
        if actual_sha256 != config.league_checkpoint_sha256:
            raise ValueError(
                f"Fixed league checkpoint SHA-256 mismatch: expected "
                f"{config.league_checkpoint_sha256}, got {actual_sha256}"
            )
        fixed_skeleton = (
            network,
            optimizer_state,
            ema_network,
            jnp.int32(0),
            jnp.int32(0),
            key,
        )
        fixed_raw, _, fixed_ema, _, _, _ = _load_checkpoint_state(
            fixed_path, fixed_skeleton, config
        )
        fixed_league_network = (
            fixed_ema if config.league_checkpoint_policy == "ema" else fixed_raw
        )

    learned_league_enabled = bool(
        config.learned_league_eval_every
        or config.learned_league_extra_eval_iterations
    )
    learned_league_members: dict[str, list[dict]] = {"raw": [], "ema": []}
    if learned_league_enabled:
        anchor_path = Path(config.learned_league_anchor_path)
        anchor_sha256 = _sha256_file(anchor_path)
        if anchor_sha256 != config.learned_league_anchor_sha256:
            raise ValueError(
                "Learned league anchor SHA-256 mismatch: expected "
                f"{config.learned_league_anchor_sha256}, got {anchor_sha256}"
            )
        if resume and Path(resume).resolve() == anchor_path.resolve():
            anchor_raw, anchor_ema = network, ema_network
            anchor_iteration = start_iteration
        else:
            anchor_skeleton = (
                network,
                optimizer_state,
                ema_network,
                jnp.int32(0),
                jnp.int32(0),
                key,
            )
            (
                anchor_raw,
                _,
                anchor_ema,
                anchor_saved_iteration,
                _,
                _,
            ) = _load_checkpoint_state(anchor_path, anchor_skeleton, config)
            anchor_iteration = int(anchor_saved_iteration)
        for policy_name, anchor_network in (
            ("raw", anchor_raw),
            ("ema", anchor_ema),
        ):
            learned_league_members[policy_name].append(
                {
                    "name": config.learned_league_anchor_name,
                    "iteration": anchor_iteration,
                    "sha256": anchor_sha256,
                    "network": anchor_network,
                }
            )

        # A recovery resume reconstructs already-admitted snapshots from the
        # numbered archives. A fresh 4,400 continuation has no such files yet.
        first_admission = (
            (anchor_iteration // config.learned_league_add_every) + 1
        ) * config.learned_league_add_every
        for snapshot_iteration in range(
            first_admission,
            start_iteration + 1,
            config.learned_league_add_every,
        ):
            snapshot_path = run_dir / f"checkpoint_{snapshot_iteration:06d}.eqx"
            if not snapshot_path.is_file():
                continue
            snapshot_skeleton = (
                network,
                optimizer_state,
                ema_network,
                jnp.int32(0),
                jnp.int32(0),
                key,
            )
            (
                snapshot_raw,
                _,
                snapshot_ema,
                loaded_iteration,
                _,
                _,
            ) = _load_checkpoint_state(snapshot_path, snapshot_skeleton, config)
            if int(loaded_iteration) != snapshot_iteration:
                raise ValueError(
                    f"League snapshot {snapshot_path} contains iteration "
                    f"{int(loaded_iteration)}"
                )
            snapshot_sha256 = _sha256_file(snapshot_path)
            for policy_name, snapshot_network in (
                ("raw", snapshot_raw),
                ("ema", snapshot_ema),
            ):
                learned_league_members[policy_name].append(
                    {
                        "name": f"continuation_{snapshot_iteration:06d}",
                        "iteration": snapshot_iteration,
                        "sha256": snapshot_sha256,
                        "network": snapshot_network,
                    }
                )
        _write_learned_league_manifest(run_dir, learned_league_members)
    (run_dir / "initialization.json").write_text(
        json.dumps(initialization, indent=2, sort_keys=True), encoding="utf-8"
    )
    initialization_metrics = {
        "iteration": start_iteration,
        "initialization/parameter_count": parameter_count,
    }
    if "conv_calibration" in initialization:
        initialization_metrics.update(
            {
                f"initialization/{name}": value
                for name, value in initialization["conv_calibration"].items()
            }
        )
    tracker.log_initialization(initialization_metrics)

    if initialization_gate and not resume:
        gate_path = Path(initialization_gate)
        print(f"Waiting for initialization approval at {gate_path}")
        while not gate_path.exists():
            if stop_at_unix is not None and time.time() >= stop_at_unix:
                raise TimeoutError("Training deadline reached while waiting for initialization gate")
            if stop_requested:
                raise RuntimeError("Stopped before initialization was approved")
            time.sleep(1)
        print("Initialization approved")
    evaluation_environment = None
    evaluation_pool = None
    evaluator = None
    if config.eval_every > 0:
        eval_pool_size = max(config.eval_games // 2, 16)
        evaluation_environment = make_environment(
            config, stage, pool_size=eval_pool_size
        )
        key, evaluation_pool_key = jax.random.split(key)
        evaluation_pool, _ = evaluation_environment.reset(evaluation_pool_key)
        evaluator = _make_evaluator(
            config, evaluation_environment, config.eval_games // 2, config.truncation
        )

    counterfactual_buffer = None
    counterfactual_root_key = None
    if config.counterfactual_castle_training:
        counterfactual_buffer = CastleCounterfactualBuffer(
            capacity=config.counterfactual_buffer_capacity,
            max_age=config.counterfactual_max_age_iterations,
            repetitions=config.counterfactual_repetitions,
            run_dir=run_dir,
        )
        counterfactual_root_key = jax.random.fold_in(
            jax.random.PRNGKey(config.seed), 0xC0A57E
        )
        if any((run_dir / "counterfactual_buffer").glob("refresh_*.npz")):
            counterfactual_buffer.load_completed_shards(start_iteration)
        counterfactual_needed_after_resume = bool(
            _linear_anneal_multiplier(
                start_iteration + 1,
                config.counterfactual_actor_anneal_start,
                config.counterfactual_actor_anneal_end,
            )
            > 0.0
            or _linear_anneal_multiplier(
                start_iteration + 1,
                config.counterfactual_critic_anneal_start,
                config.counterfactual_critic_anneal_end,
            )
            > 0.0
        )
        if counterfactual_buffer.size == 0 and counterfactual_needed_after_resume:
            initial_generation_started = time.perf_counter()
            refresh, refresh_metrics = generate_counterfactual_refresh(
                network=network,
                pool_replicated=pool_replicated,
                key=jax.random.fold_in(counterfactual_root_key, start_iteration),
                config=config,
                environment=environment,
                iteration=start_iteration,
            )
            shard_path = counterfactual_buffer.add(refresh, start_iteration)
            refresh_metrics.update(
                {
                    "iteration": start_iteration,
                    "counterfactual/ready_iteration": start_iteration,
                    "counterfactual/buffer_size": counterfactual_buffer.size,
                    "counterfactual/generation_seconds": time.perf_counter()
                    - initial_generation_started,
                    "counterfactual/shard": str(shard_path),
                }
            )
            _write_metrics(metrics_path, refresh_metrics)
            tracker.log_evaluation(refresh_metrics)
            print(
                "Initial counterfactual buffer ready: "
                f"{counterfactual_buffer.size} candidates in "
                f"{refresh_metrics['counterfactual/generation_seconds']:.1f}s"
            )
        elif counterfactual_buffer.size:
            print(
                "Restored counterfactual buffer: "
                f"{counterfactual_buffer.size} candidates through iteration "
                f"{start_iteration}"
            )
        else:
            print(
                "Counterfactual schedules are complete; continuing without a "
                "counterfactual replay buffer"
            )
        _write_json_atomic(
            run_dir / "counterfactual_rng.json",
            {
                "scheme": "jax.random.fold_in(root, raw_generator_iteration)",
                "root": np.asarray(jax.device_get(counterfactual_root_key)).tolist(),
                "ppo_rng_untouched": True,
            },
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
        indices = jax.random.randint(rng, (config.num_envs,), 0, environment.pool_size)
        sampled = jax.tree.map(lambda value: value[indices], pool_shard)
        return sampled._replace(pool_idx=indices.astype(jnp.int32))

    sample_initial_states_pmapped = jax.pmap(sample_initial_states)
    key, states_key = jax.random.split(key)
    device_keys = jax.random.split(states_key, jax.device_count())
    states = sample_initial_states_pmapped(pool_replicated, device_keys)

    per_device_memory = _batched_memory(config)
    memory_zero = _replicate_for_pmap(per_device_memory)
    memory_one = _replicate_for_pmap(per_device_memory)
    per_device_castle_flags = jnp.zeros((config.num_envs,), dtype=jnp.int32)
    castle_flags_zero = _replicate_for_pmap(per_device_castle_flags)
    castle_flags_one = _replicate_for_pmap(per_device_castle_flags)
    key, rollout_key = jax.random.split(key)
    rollout_keys = jax.random.split(rollout_key, jax.device_count())

    def rollout_shard(
        params,
        states,
        rng,
        mem_zero,
        mem_one,
        castle_zero,
        castle_one,
        pool_shard,
        potential_shaping_scale,
        tactical_build_logit_boost,
    ):
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
            castle_flags_player_zero=castle_zero,
            castle_flags_player_one=castle_one,
            return_castle_state=True,
            observation_schema=config.observation_schema,
            potential_shaping_scale=potential_shaping_scale,
            tactical_build_logit_boost=tactical_build_logit_boost,
            tactical_build_post_reserve=config.tactical_build_post_reserve,
            tactical_build_payback_margin=config.tactical_build_payback_margin,
            tactical_build_remembered_enemy_turns=(
                config.tactical_build_remembered_enemy_turns
            ),
            tactical_build_enemy_safety_radius=(
                config.tactical_build_enemy_safety_radius
            ),
        )

    rollout_pmapped = jax.pmap(rollout_shard)

    kept_samples = int(
        2 * config.num_envs * config.num_steps * config.advantage_top_fraction
    )
    prepare_batch = make_prepare_batch(config)
    natural_castle_metrics = make_natural_castle_metrics(config)
    ema_step = make_ema_step(config)
    ppo_update_shard = make_update_shard(config, static, optimizer)
    counterfactual_update_shard = (
        make_counterfactual_update_shard(config, static, optimizer)
        if config.counterfactual_castle_training
        else None
    )

    device_count = jax.device_count()
    train_started = time.perf_counter()
    if duration_seconds is not None:
        stop_at_unix = time.time() + duration_seconds
    completed_iterations = start_iteration
    latest_every = config.latest_checkpoint_every or config.checkpoint_every
    periodic_raw_enabled = "raw" in config.league_eval_policies
    last_league_finished_at = train_started
    cumulative_league_seconds = 0.0
    publish_status_path = run_dir / "publish_status.jsonl"
    publish_status_offset = 0
    trace_active = False
    trace_stop_iteration = trace_start_iteration + trace_iterations - 1
    castle_window_counts: dict[str, float] = {}
    for iteration in range(start_iteration, config.num_iterations):
        if stop_requested or (stop_at_unix is not None and time.time() >= stop_at_unix):
            print("Training soft deadline reached before starting another iteration")
            break
        iteration_number = iteration + 1
        actor_schedule_scale = _linear_anneal_multiplier(
            iteration_number,
            config.counterfactual_actor_anneal_start,
            config.counterfactual_actor_anneal_end,
        )
        critic_schedule_scale = _linear_anneal_multiplier(
            iteration_number,
            config.counterfactual_critic_anneal_start,
            config.counterfactual_critic_anneal_end,
        )
        counterfactual_active = bool(
            config.counterfactual_castle_training
            and (actor_schedule_scale > 0.0 or critic_schedule_scale > 0.0)
        )
        cadence_iteration = (
            iteration_number - start_iteration
            if config.cadence_relative_to_resume
            else iteration_number
        )
        if trace_iterations and iteration_number == trace_start_iteration:
            Path(trace_dir).mkdir(parents=True, exist_ok=True)
            # Python call tracing is already covered by the isolated cProfile
            # pass. Keeping it disabled here substantially reduces XPlane
            # trace volume while preserving device kernels and lightweight
            # host/XLA launch events.
            profile_options = jax.profiler.ProfileOptions()
            profile_options.host_tracer_level = 1
            profile_options.python_tracer_level = 0
            profile_options.enable_hlo_proto = False
            jax.profiler.start_trace(trace_dir, profiler_options=profile_options)
            trace_active = True
            print(
                f"JAX trace started at iteration {iteration_number}; "
                f"capturing {trace_iterations} iterations to {trace_dir}"
            )
        iteration_started = time.perf_counter()
        intervention_strength = _castle_intervention_strength(
            config, iteration_number
        )
        potential_shaping_scale = (
            intervention_strength * config.castle_potential_scale
            if config.actor_potential_shaping
            else 0.0
        )
        tactical_build_logit_boost = (
            intervention_strength * config.tactical_build_logit_boost
        )
        potential_scale_by_device = np.full(
            (device_count,), potential_shaping_scale, dtype=np.float32
        )
        build_boost_by_device = np.full(
            (device_count,), tactical_build_logit_boost, dtype=np.float32
        )
        rollout_started = time.perf_counter()
        (
            states,
            rollout,
            rollout_keys,
            memory_zero,
            memory_one,
            castle_flags_zero,
            castle_flags_one,
        ) = rollout_pmapped(
            parameters,
            states,
            rollout_keys,
            memory_zero,
            memory_one,
            castle_flags_zero,
            castle_flags_one,
            pool_replicated,
            potential_scale_by_device,
            build_boost_by_device,
        )
        if config.debug_timing:
            jax.block_until_ready(states)
        rollout_seconds = time.perf_counter() - rollout_started
        (
            observations,
            histories,
            masks,
            tactical_build_masks,
            actions,
            old_log_probs,
            values,
            next_values,
            rewards,
            actor_rewards,
            potential_rewards,
            terminated,
            truncated,
            winners,
            base_policy_statistics,
            behavior_policy_statistics,
            selected_tactical_build,
            build_actions,
            legal_build_opportunities,
            completed_game_with_build,
            completed_player_game_with_build,
            completed_player_games,
            completed_game_builds,
        ) = rollout

        advantages, returns, sample_indices, prep_metrics = prepare_batch(
            rewards,
            actor_rewards,
            potential_rewards,
            values,
            next_values,
            terminated,
            truncated,
            winners,
            actions,
            tactical_build_masks,
            selected_tactical_build,
            base_policy_statistics,
            behavior_policy_statistics,
        )
        castle_metrics = natural_castle_metrics(
            build_actions,
            legal_build_opportunities,
            completed_game_with_build,
            completed_player_game_with_build,
            completed_player_games,
            completed_game_builds,
            terminated,
            truncated,
        )
        batch = (
            observations,
            histories,
            masks,
            actions,
            old_log_probs,
            advantages,
            returns,
            tactical_build_masks,
        )

        counterfactual_epoch = None
        counterfactual_epoch_metrics = {}
        counterfactual_actor_scale_by_device = None
        counterfactual_critic_scale_by_device = None
        if counterfactual_active:
            optimizer_minibatches = kept_samples // config.minibatch_size
            sampled = counterfactual_buffer.sample_epoch(
                current_iteration=iteration,
                device_count=device_count,
                minibatches=optimizer_minibatches,
                seed=config.seed * 1_000_003 + iteration_number,
                recent_fraction=config.counterfactual_recent_fraction,
                actor_uniform_per_device=(
                    config.counterfactual_actor_uniform_per_device
                ),
                actor_positive_per_device=(
                    config.counterfactual_actor_positive_per_device
                ),
                actor_negative_per_device=(
                    config.counterfactual_actor_negative_per_device
                ),
                successor_per_device=(
                    config.counterfactual_successor_minibatch_size_per_device
                ),
            )
            counterfactual_epoch = jax.tree.map(
                jnp.asarray,
                {
                    "actor_cache": sampled["actor_cache"],
                    "actor_indices": sampled["actor_indices"],
                    "actor_inverse_probability": sampled[
                        "actor_inverse_probability"
                    ],
                    "successor_cache": sampled["successor_cache"],
                    "successor_indices": sampled["successor_indices"],
                    "gradient_diagnostics_enabled": np.full(
                        (device_count,),
                        (iteration_number - start_iteration == 1)
                        or (
                            (iteration_number - start_iteration)
                            % config.counterfactual_refresh_every
                            == 0
                        ),
                        dtype=np.bool_,
                    ),
                },
            )
            coefficient_scale = min(
                1.0,
                counterfactual_buffer.size
                / config.counterfactual_unique_examples_full_weight,
            )
            counterfactual_actor_scale_by_device = np.full(
                (device_count,),
                coefficient_scale * actor_schedule_scale,
                dtype=np.float32,
            )
            counterfactual_critic_scale_by_device = np.full(
                (device_count,),
                coefficient_scale * critic_schedule_scale,
                dtype=np.float32,
            )
            counterfactual_epoch_metrics = {
                "counterfactual/buffer_size": counterfactual_buffer.size,
                "counterfactual/positive_examples": sampled["positive_examples"],
                "counterfactual/negative_examples": sampled["negative_examples"],
                "counterfactual/uncertain_examples": sampled[
                    "uncertain_examples"
                ],
                "counterfactual/generator_lag": iteration
                - sampled["latest_generator_iteration"],
                "counterfactual/coefficient_scale": coefficient_scale,
                "counterfactual/actor_schedule_scale": actor_schedule_scale,
                "counterfactual/critic_schedule_scale": critic_schedule_scale,
                "counterfactual/actor_total_scale": coefficient_scale
                * actor_schedule_scale,
                "counterfactual/critic_total_scale": coefficient_scale
                * critic_schedule_scale,
            }
        elif config.counterfactual_castle_training:
            counterfactual_epoch_metrics = {
                "counterfactual/actor_schedule_scale": actor_schedule_scale,
                "counterfactual/critic_schedule_scale": critic_schedule_scale,
                "counterfactual/actor_total_scale": 0.0,
                "counterfactual/critic_total_scale": 0.0,
            }

        update_started = time.perf_counter()
        entropy_coefficient = _entropy_coefficient(config, iteration)
        entropy_by_device = np.full(
            (device_count,), entropy_coefficient, dtype=np.float32
        )
        epochs_used = 0
        metrics = None
        for _ in range(config.ppo_epochs):
            if counterfactual_active:
                (
                    parameters,
                    optimizer_state,
                    rollout_keys,
                    metrics,
                ) = counterfactual_update_shard(
                    parameters,
                    optimizer_state,
                    batch,
                    sample_indices,
                    counterfactual_epoch,
                    rollout_keys,
                    entropy_by_device,
                    counterfactual_actor_scale_by_device,
                    counterfactual_critic_scale_by_device,
                )
            else:
                (
                    parameters,
                    optimizer_state,
                    rollout_keys,
                    metrics,
                ) = ppo_update_shard(
                    parameters,
                    optimizer_state,
                    batch,
                    sample_indices,
                    rollout_keys,
                    entropy_by_device,
                    build_boost_by_device,
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
        host_metrics = jax.device_get({**prep_metrics, **castle_metrics, **metrics})
        host_metrics = {name: value[0] for name, value in host_metrics.items()}
        episodes = int(host_metrics.pop("episodes"))
        wins = int(host_metrics.pop("wins"))
        losses = int(host_metrics.pop("losses"))
        draws = episodes - wins - losses
        explained_variance = float(host_metrics.pop("explained_variance"))
        per_iteration_castle_counts = {
            name: float(value)
            for name, value in host_metrics.items()
            if name.startswith("ppo_castle/count/")
        }
        per_iteration_castle_rates = _natural_castle_rates(
            per_iteration_castle_counts
        )
        for name, value in per_iteration_castle_counts.items():
            castle_window_counts[name] = castle_window_counts.get(name, 0.0) + value

        iteration_seconds = time.perf_counter() - iteration_started
        total_samples = device_count * 2 * config.num_envs * config.num_steps
        samples_per_second = total_samples / iteration_seconds
        completed_iterations = iteration + 1
        cumulative_samples = completed_iterations * total_samples
        elapsed_training = time.perf_counter() - train_started
        current_learning_rate = _learning_rate_float(
            config,
            iteration * (config.ppo_epochs * kept_samples // config.minibatch_size),
        )
        record = {
            "iteration": iteration + 1,
            "wall_seconds": elapsed_training,
            "active_training_seconds": elapsed_training,
            "iteration_seconds": iteration_seconds,
            "stage": stage_index,
            "cumulative_samples": cumulative_samples,
            "continuation_iteration": completed_iterations - start_iteration,
            "continuation_samples": (completed_iterations - start_iteration)
            * total_samples,
            "samples_per_second": samples_per_second,
            "samples_per_gpu_second": samples_per_second / device_count,
            "allocated_gpu_hours": elapsed_training * device_count / 3600.0,
            "training_gpu_hours": elapsed_training * device_count / 3600.0,
            "episodes": episodes,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "score": (wins + 0.5 * draws) / max(episodes, 1),
            "explained_variance": explained_variance,
            "learning_rate": current_learning_rate,
            "entropy_coefficient": entropy_coefficient,
            "castle_intervention_strength": intervention_strength,
            "castle_potential_shaping_scale": potential_shaping_scale,
            "castle_tactical_logit_boost": tactical_build_logit_boost,
            "epochs_used": epochs_used,
            **counterfactual_epoch_metrics,
            **per_iteration_castle_rates,
            **{name: float(value) for name, value in host_metrics.items()},
        }
        if config.debug_timing:
            record["rollout_seconds"] = rollout_seconds
            record["update_seconds"] = update_seconds
            record["host_seconds"] = max(
                0.0, iteration_seconds - rollout_seconds - update_seconds
            )

        if (iteration_number - start_iteration) % 50 == 0:
            window_record = {
                "iteration": iteration_number,
                **{
                    name.replace("ppo_castle/count/", "ppo_castle/window50/count/"): value
                    for name, value in castle_window_counts.items()
                },
                **_natural_castle_rates(
                    castle_window_counts, prefix="ppo_castle/window50/"
                ),
            }
            _write_metrics(metrics_path, window_record)
            tracker.log_evaluation(window_record)
            castle_window_counts = {}

        if cadence_iteration % config.metrics_every == 0:
            _write_metrics(metrics_path, record)
            tracker.log_training(record)
            print(
                f"iter {iteration + 1:6d} | loss {record['loss']:.4f} | "
                f"entropy {record['entropy']:.3f} | KL {record['approximate_kl']:.4f} | "
                f"episodes {episodes} W/L/D {wins}/{losses}/{draws} | "
                f"EV {explained_variance:.2f} | {samples_per_second:,.0f} samples/s"
            )

        # Release the large PPO rollout and replay-cache views before periodic
        # paired generation or league evaluation allocates its own trajectories.
        del rollout, batch, observations, histories, masks, actions, old_log_probs
        del tactical_build_masks, actor_rewards, potential_rewards
        del base_policy_statistics, behavior_policy_statistics
        del selected_tactical_build
        del build_actions, legal_build_opportunities
        del completed_game_with_build, completed_player_game_with_build
        del completed_player_games, completed_game_builds
        del values, next_values, rewards, terminated, truncated, winners
        del advantages, returns, sample_indices
        if counterfactual_epoch is not None:
            del counterfactual_epoch

        should_refresh_counterfactual = (
            counterfactual_active
            and (iteration_number - start_iteration)
            % config.counterfactual_refresh_every
            == 0
            and iteration_number < config.num_iterations
        )
        if should_refresh_counterfactual:
            generation_started = time.perf_counter()
            generator_network = eqx.combine(
                _first_pmap_replica_to_host(parameters), static
            )
            refresh, refresh_metrics = generate_counterfactual_refresh(
                network=generator_network,
                pool_replicated=pool_replicated,
                key=jax.random.fold_in(counterfactual_root_key, iteration_number),
                config=config,
                environment=environment,
                iteration=iteration_number,
            )
            shard_path = counterfactual_buffer.add(refresh, iteration_number)
            generation_record = {
                "iteration": iteration_number,
                **refresh_metrics,
                "counterfactual/buffer_size": counterfactual_buffer.size,
                "counterfactual/generation_seconds": time.perf_counter()
                - generation_started,
                "counterfactual/shard": str(shard_path),
            }
            _write_metrics(metrics_path, generation_record)
            tracker.log_evaluation(generation_record)
            print(
                "  counterfactual refresh: "
                f"{len(refresh['build_action'])} candidates, "
                f"buffer={counterfactual_buffer.size}, "
                f"{generation_record['counterfactual/generation_seconds']:.1f}s"
            )

        should_evaluate = config.eval_every > 0 and (
            cadence_iteration % config.eval_every == 0
        )
        if should_evaluate:
            evaluation_started = time.perf_counter()
            stage_before = stage_index
            key, evaluation_key = jax.random.split(key)
            ema_network = eqx.combine(
                _first_pmap_replica_to_host(ema_parameters), static
            )
            evaluation, _ = evaluator(evaluation_pool, ema_network, evaluation_key)
            evaluation = jax.tree.map(float, evaluation)
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
                    pool_replicated = _reset_replicated_pool(environment, pool_key)
                    rollout_pmapped = jax.pmap(rollout_shard)
                    key, states_key = jax.random.split(key)
                    device_keys = jax.random.split(states_key, jax.device_count())
                    states = sample_initial_states_pmapped(pool_replicated, device_keys)
                    memory_zero = _replicate_for_pmap(per_device_memory)
                    memory_one = _replicate_for_pmap(per_device_memory)
                    castle_flags_zero = _replicate_for_pmap(
                        per_device_castle_flags
                    )
                    castle_flags_one = _replicate_for_pmap(
                        per_device_castle_flags
                    )

                    evaluation_environment = make_environment(
                        config, stage, pool_size=eval_pool_size
                    )
                    key, evaluation_pool_key = jax.random.split(key)
                    evaluation_pool, _ = evaluation_environment.reset(
                        evaluation_pool_key
                    )
                    evaluator = _make_evaluator(
                        config,
                        evaluation_environment,
                        config.eval_games // 2,
                        config.truncation,
                    )

            evaluation_record = {
                "iteration": iteration + 1,
                "curriculum_eval/evaluation_seconds": time.perf_counter()
                - evaluation_started,
                "curriculum_eval/stage_before": stage_before,
                "curriculum_eval/stage_after": stage_index,
                "curriculum_eval/advanced": int(stage_index != stage_before),
                "curriculum_eval/ema/games": evaluation["wins"]
                + evaluation["losses"]
                + evaluation["draws"],
                **{
                    f"curriculum_eval/ema/{name}": value
                    for name, value in evaluation.items()
                },
            }
            _write_metrics(metrics_path, evaluation_record)
            tracker.log_evaluation(evaluation_record)

        should_archive = (
            cadence_iteration % config.checkpoint_every == 0
            or iteration_number in config.checkpoint_extra_iterations
        )
        should_save_latest = cadence_iteration % latest_every == 0
        if should_archive or should_save_latest:
            checkpoint_started = time.perf_counter()
            checkpoint_metadata = _save_periodic_checkpoint(
                run_dir,
                parameters,
                static,
                optimizer_state,
                ema_parameters,
                iteration + 1,
                stage_index,
                key,
                archive=should_archive,
            )
            print(
                f"  saved {checkpoint_metadata['path']} "
                f"({checkpoint_metadata['sha256']})"
            )
            timing_record = {
                "iteration": iteration + 1,
                "checkpoint/latest_saved": 1,
                "checkpoint/archive_saved": int(should_archive),
                "checkpoint/bytes": checkpoint_metadata["bytes"],
                "checkpoint/sha256": checkpoint_metadata["sha256"],
                "checkpoint/raw_weights_present": 1,
                "checkpoint/optimizer_state_present": 1,
                "checkpoint/ema_weights_present": 1,
                "checkpoint/save_seconds": time.perf_counter()
                - checkpoint_started,
            }
            if config.counterfactual_castle_training:
                buffer_manifest = (
                    run_dir / "counterfactual_buffer" / "manifest.json"
                )
                counterfactual_checkpoint_state = {
                    "schema": "castle_counterfactual_checkpoint_sidecar",
                    "schema_version": config.counterfactual_schema_version,
                    "checkpoint_iteration": iteration + 1,
                    "buffer_size": counterfactual_buffer.size,
                    "buffer_manifest": str(buffer_manifest),
                    "latest_generator_iteration": (
                        _latest_counterfactual_generator_iteration(
                            counterfactual_buffer
                        )
                    ),
                    "counterfactual_ready_iteration": start_iteration,
                }
                sidecar_name = (
                    f"checkpoint_{iteration + 1:06d}.counterfactual.json"
                    if should_archive
                    else "latest.counterfactual.json"
                )
                _write_json_atomic(
                    run_dir / sidecar_name, counterfactual_checkpoint_state
                )
                timing_record["checkpoint/counterfactual_state_present"] = 1
            if should_archive:
                publication = _request_checkpoint_publication(
                    run_dir, checkpoint_metadata
                )
                timing_record["checkpoint/hf_export_requested"] = 1
                tracker.log_checkpoint_export(publication)
            _write_metrics(metrics_path, timing_record)
            tracker.log_evaluation(timing_record)

        should_run_league = config.league_eval_every > 0 and (
            cadence_iteration % config.league_eval_every == 0
        )
        if should_run_league:
            league_started = time.perf_counter()
            current_network = eqx.combine(
                _first_pmap_replica_to_host(parameters), static
            )
            ema_network = eqx.combine(
                _first_pmap_replica_to_host(ema_parameters), static
            )
            policy_names = [
                name
                for name in config.league_eval_policies
                if name != "raw" or periodic_raw_enabled
            ]
            policy_networks = {
                name: current_network if name == "raw" else ema_network
                for name in policy_names
            }
            _run_league(
                config,
                policy_networks,
                tracker,
                run_dir,
                iteration + 1,
                label=f"{iteration + 1:06d}",
                checkpoint_opponent=fixed_league_network,
            )
            league_finished = time.perf_counter()
            league_seconds = league_finished - league_started
            cumulative_league_seconds += league_seconds
            interval_seconds = max(league_started - last_league_finished_at, 1e-9)
            interval_overhead_fraction = league_seconds / (
                interval_seconds + league_seconds
            )
            run_time_fraction = cumulative_league_seconds / max(
                league_finished - train_started, 1e-9
            )
            overhead_record = {
                "iteration": iteration + 1,
                "league/periodic/evaluation_seconds": league_seconds,
                "league/periodic/cumulative_evaluation_seconds": cumulative_league_seconds,
                "league/periodic/interval_overhead_fraction": interval_overhead_fraction,
                "league/periodic/run_time_fraction": run_time_fraction,
                "league/periodic/raw_enabled": int(periodic_raw_enabled),
            }
            _write_metrics(metrics_path, overhead_record)
            tracker.log_evaluation(overhead_record)
            if (
                periodic_raw_enabled
                and interval_overhead_fraction
                > config.league_periodic_raw_max_overhead_fraction
            ):
                periodic_raw_enabled = False
                print(
                    "  periodic league overhead exceeded threshold; "
                    "future periodic evaluations will use EMA only"
                )
            last_league_finished_at = league_finished

        should_run_learned_league = learned_league_enabled and (
            (
                config.learned_league_eval_every > 0
                and iteration_number % config.learned_league_eval_every == 0
            )
            or iteration_number in config.learned_league_extra_eval_iterations
        )
        if should_run_learned_league:
            learned_current_raw = eqx.combine(
                _first_pmap_replica_to_host(parameters), static
            )
            learned_current_ema = eqx.combine(
                _first_pmap_replica_to_host(ema_parameters), static
            )
            _run_learned_checkpoint_league(
                config,
                {"raw": learned_current_raw, "ema": learned_current_ema},
                learned_league_members,
                tracker,
                run_dir,
                iteration_number,
            )
            if iteration_number % config.learned_league_add_every == 0:
                archive_path = run_dir / f"checkpoint_{iteration_number:06d}.eqx"
                if not archive_path.is_file():
                    raise FileNotFoundError(
                        "A growing-league admission requires its numbered training "
                        f"checkpoint: {archive_path}"
                    )
                archive_sha256 = _sha256_file(archive_path)
                member_name = f"continuation_{iteration_number:06d}"
                for policy_name, policy_network in (
                    ("raw", learned_current_raw),
                    ("ema", learned_current_ema),
                ):
                    learned_league_members[policy_name].append(
                        {
                            "name": member_name,
                            "iteration": iteration_number,
                            "sha256": archive_sha256,
                            "network": policy_network,
                        }
                    )
                _write_learned_league_manifest(run_dir, learned_league_members)
                admission_record = {
                    "iteration": iteration_number,
                    "learned_league/admitted": 1,
                    "learned_league/member_count": len(
                        learned_league_members["raw"]
                    ),
                    "learned_league/admitted_checkpoint_sha256": archive_sha256,
                }
                _write_metrics(metrics_path, admission_record)
                tracker.log_evaluation(admission_record)

        publish_status_offset = _ingest_publish_status(
            publish_status_path, publish_status_offset, tracker, metrics_path
        )

        if (
            config.reset_pool_every > 0
            and cadence_iteration % config.reset_pool_every == 0
        ):
            pool_reset_started = time.perf_counter()
            key, pool_key = jax.random.split(key)
            pool_replicated = _reset_replicated_pool(environment, pool_key)
            if config.debug_timing:
                timing_record = {
                    "iteration": iteration + 1,
                    "performance/pool_reset_seconds": time.perf_counter()
                    - pool_reset_started,
                }
                _write_metrics(metrics_path, timing_record)
                tracker.log_evaluation(timing_record)

        if trace_active and iteration_number == trace_stop_iteration:
            jax.block_until_ready(parameters)
            jax.profiler.stop_trace()
            trace_active = False
            print(f"JAX trace stopped after iteration {iteration_number}")

        if stop_requested or (stop_at_unix is not None and time.time() >= stop_at_unix):
            print(f"Stopping cleanly after iteration {completed_iterations}")
            break

    if trace_active:
        jax.block_until_ready(parameters)
        jax.profiler.stop_trace()

    final_network = eqx.combine(_first_pmap_replica_to_host(parameters), static)
    final_optimizer_state = _first_pmap_replica_to_host(optimizer_state)
    final_ema_network = eqx.combine(
        _first_pmap_replica_to_host(ema_parameters), static
    )
    terminal_checkpoint_started = time.perf_counter()
    final_metadata = _save_checkpoint(
        run_dir / "terminal.eqx",
        final_network,
        final_optimizer_state,
        final_ema_network,
        completed_iterations,
        stage_index,
        key,
    )
    _copy_atomic(run_dir / "terminal.eqx", run_dir / "final.eqx")
    _copy_atomic(run_dir / "terminal.eqx", run_dir / "latest.eqx")
    final_metadata.update({"iteration": completed_iterations, "stage": stage_index})
    (run_dir / "terminal_checkpoint.json").write_text(
        json.dumps(final_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    if config.counterfactual_castle_training:
        _write_json_atomic(
            run_dir / "terminal.counterfactual.json",
            {
                "schema": "castle_counterfactual_checkpoint_sidecar",
                "schema_version": config.counterfactual_schema_version,
                "checkpoint_iteration": completed_iterations,
                "buffer_size": counterfactual_buffer.size,
                "buffer_manifest": str(
                    run_dir / "counterfactual_buffer" / "manifest.json"
                ),
                "latest_generator_iteration": (
                    _latest_counterfactual_generator_iteration(
                        counterfactual_buffer
                    )
                ),
                "counterfactual_ready_iteration": start_iteration,
            },
        )
    tracker.log_evaluation(
        {
            "iteration": completed_iterations,
            "checkpoint/terminal_saved": 1,
            "checkpoint/bytes": final_metadata["bytes"],
            "checkpoint/sha256": final_metadata["sha256"],
            "checkpoint/raw_weights_present": 1,
            "checkpoint/optimizer_state_present": 1,
            "checkpoint/ema_weights_present": 1,
            "checkpoint/save_seconds": time.perf_counter()
            - terminal_checkpoint_started,
            "checkpoint/hf_export_requested": 1,
        }
    )
    terminal_publication = _request_checkpoint_publication(run_dir, final_metadata)
    tracker.log_checkpoint_export(terminal_publication)
    if config.terminal_raw_ema_head_to_head_maps > 0:
        terminal_head_to_head_started = time.perf_counter()
        terminal_head_to_head = _evaluate_network_pair(
            config,
            final_network,
            final_ema_network,
            n_maps=config.terminal_raw_ema_head_to_head_maps,
            seed=config.learned_league_seed + 1,
        )
        terminal_head_to_head["evaluation_seconds"] = (
            time.perf_counter() - terminal_head_to_head_started
        )
        terminal_payload = {
            "schema": "terminal_raw_vs_ema_paired_head_to_head",
            "iteration": completed_iterations,
            "seed": config.learned_league_seed + 1,
            "maps": config.terminal_raw_ema_head_to_head_maps,
            "games": 2 * config.terminal_raw_ema_head_to_head_maps,
            "perspective": "raw_as_policy_a_vs_ema_as_policy_b",
            "result": terminal_head_to_head,
        }
        _write_json_atomic(run_dir / "terminal_raw_vs_ema.json", terminal_payload)
        terminal_record = {
            "iteration": completed_iterations,
            **{
                f"terminal_raw_vs_ema/{name}": value
                for name, value in terminal_head_to_head.items()
            },
        }
        _write_metrics(metrics_path, terminal_record)
        tracker.log_evaluation(terminal_record)
    if config.league_eval_after_training:
        terminal_policies = {
            name: final_network if name == "raw" else final_ema_network
            for name in config.league_eval_policies
        }
        _run_league(
            config,
            terminal_policies,
            tracker,
            run_dir,
            completed_iterations,
            label="final",
            checkpoint_opponent=fixed_league_network,
        )
    publish_status_offset = _ingest_publish_status(
        publish_status_path, publish_status_offset, tracker, metrics_path
    )
    elapsed = time.perf_counter() - train_started
    tracker.update_summary(
        {
            "final_iteration": completed_iterations,
            "final_curriculum_stage": stage_index,
            "final_checkpoint_sha256": final_metadata["sha256"],
            "wall_seconds": elapsed,
            "allocated_gpu_hours": elapsed * device_count / 3600.0,
        }
    )
    tracker.finish()
    for signum, handler in previous_signal_handlers.items():
        signal.signal(signum, handler)
    return final_network, final_optimizer_state, final_ema_network


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="generals/training/configs/competition_l7.toml",
        help="Path to a TOML TrainingConfig",
    )
    parser.add_argument(
        "--resume", help="Path to a checkpoint produced by this trainer"
    )
    parser.add_argument(
        "--trace-dir", help="Optional JAX/XPlane trace output directory"
    )
    parser.add_argument(
        "--trace-start-iteration",
        type=int,
        default=1,
        help="One-based iteration at which tracing starts",
    )
    parser.add_argument(
        "--trace-iterations",
        type=int,
        default=0,
        help="Number of iterations to capture; zero disables tracing",
    )
    parser.add_argument(
        "--stop-at-unix",
        type=float,
        help="Soft wall-clock deadline; finish the current iteration and checkpoint",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        help="Training-loop duration in hours, starting immediately before iteration work",
    )
    parser.add_argument(
        "--initialization-gate",
        help="For fresh runs, wait for this supervisor approval file before iteration 1",
    )
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
    train(
        config,
        resume=args.resume,
        trace_dir=args.trace_dir,
        trace_start_iteration=args.trace_start_iteration,
        trace_iterations=args.trace_iterations,
        stop_at_unix=args.stop_at_unix,
        duration_seconds=(
            args.duration_hours * 3600.0 if args.duration_hours is not None else None
        ),
        initialization_gate=args.initialization_gate,
        graceful_signals=True,
    )


if __name__ == "__main__":
    main()

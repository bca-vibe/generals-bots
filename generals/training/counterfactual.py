"""Castle counterfactual generation, replay, and auditable sampling.

Forced branches in this module are supervised intervention data only. Nothing
here manufactures behavior log-probabilities, GAE values, or PPO ratios.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from generals.core.game import get_observation

from .actions import (
    CELL_COUNT,
    MOVE_PLANES,
    PASS_INDEX,
    build_cost_grid_from_observation,
    decode_action,
    legal_action_mask,
)
from .observation import augment_observation, init_observation_memory, temporal_input

SCHEMA_NAME = "castle_counterfactual_buffer"
SCHEMA_VERSION = 1
BUILD_START = MOVE_PLANES * CELL_COUNT
TURN_BIN_WIDTH = 200
TURN_BINS = 6
BRANCH_CONTROL = 0
BRANCH_BUILD = 1

ACTOR_FIELDS = (
    "pre_observation",
    "pre_history",
    "pre_mask",
    "build_action",
    "control_action",
    "delta_mean",
    "delta_se",
    "delta_shrunk",
    "source_id",
    "source_game_id",
    "source_turn",
    "source_seat",
    "source_quota",
    "source_selection_probability",
    "proposal_method",
    "proposal_probability",
    "generator_iteration",
)

SUCCESSOR_FIELDS = (
    "successor_observation",
    "successor_history",
    "successor_mask",
    "successor_target",
    "successor_finished",
)

ACTOR_TRAIN_FIELDS = (
    "pre_observation",
    "pre_history",
    "pre_mask",
    "build_action",
    "control_action",
    "delta_mean",
    "delta_shrunk",
)

SUCCESSOR_TRAIN_FIELDS = (
    "successor_observation",
    "successor_history",
    "successor_mask",
    "successor_target",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_take(tree, indices):
    return jax.tree.map(lambda value: value[indices], tree)


def _tree_concat(trees):
    return jax.tree.map(lambda *values: np.concatenate(values), *trees)


def _tree_flatten_slots(tree):
    return jax.tree.map(
        lambda value: np.asarray(jax.device_get(value)).reshape(
            (-1, *value.shape[3:])
        ),
        tree,
    )


def _batched_memory(config, games: int):
    memory = init_observation_memory(
        config.pad_to, config.history_size, config.temporal_window
    )
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (games, *value.shape)), memory
    )


def _slot_broadcast(tree, games: int):
    return jax.tree.map(
        lambda value: jnp.broadcast_to(
            value[:, None, None], (games, 2, TURN_BINS, *value.shape[1:])
        ),
        tree,
    )


def _replace_slots(selected, source, replace):
    def replace_leaf(old, new):
        expanded = new[:, None, None]
        condition = replace.reshape(
            replace.shape + (1,) * (expanded.ndim - replace.ndim)
        )
        return jnp.where(condition, expanded, old)

    return jax.tree.map(replace_leaf, selected, source)


@eqx.filter_jit
def collect_grouped_opportunities(network, pool, key, config, environment):
    """Keep one uniform state per game/seat/200-turn group.

    This sampler is run from a frozen raw snapshot at refresh time. It mirrors
    the ordinary stochastic self-play distribution without copying every PPO
    rollout state to the host.
    """
    games = pool.armies.shape[0]
    states = pool._replace(pool_idx=jnp.arange(games, dtype=jnp.int32))
    memory_zero = _batched_memory(config, games)
    memory_one = _batched_memory(config, games)
    sampled_states = _slot_broadcast(states, games)
    sampled_memory_zero = _slot_broadcast(memory_zero, games)
    sampled_memory_one = _slot_broadcast(memory_one, games)
    seen = jnp.zeros((games, 2, TURN_BINS), dtype=jnp.int32)
    found = jnp.zeros((games, 2, TURN_BINS), dtype=jnp.bool_)
    finished = jnp.zeros((games,), dtype=jnp.bool_)

    def step(carry, _):
        (
            states,
            rng,
            memory_zero,
            memory_one,
            sampled_states,
            sampled_memory_zero,
            sampled_memory_one,
            seen,
            found,
            finished,
        ) = carry
        prior_memory_zero, prior_memory_one = memory_zero, memory_one
        obs_zero = jax.vmap(lambda state: get_observation(state, 0))(states)
        obs_one = jax.vmap(lambda state: get_observation(state, 1))(states)
        observations = jax.tree.map(
            lambda zero, one: jnp.concatenate([zero, one]), obs_zero, obs_one
        )
        memories = jax.tree.map(
            lambda zero, one: jnp.concatenate([zero, one]),
            memory_zero,
            memory_one,
        )
        board_masks = jnp.concatenate([states.board_mask, states.board_mask])
        augmented, memories = jax.vmap(
            lambda observation, memory, board_mask: augment_observation(
                observation,
                memory,
                board_mask,
                config.observation_schema,
                environment.deathtouch_turn or 800,
            )
        )(observations, memories, board_masks)
        masks = jax.vmap(legal_action_mask)(observations, board_masks)
        histories = temporal_input(memories)
        logits = jax.vmap(network.forward)(augmented, histories, masks)[0]

        masks_by_seat = jnp.stack([masks[:games], masks[games:]], axis=1)
        opportunity = (~finished[:, None]) & jnp.any(
            masks_by_seat[:, :, BUILD_START:PASS_INDEX], axis=-1
        )
        turn_bin = jnp.minimum(states.time // TURN_BIN_WIDTH, TURN_BINS - 1)
        in_bin = jax.nn.one_hot(turn_bin, TURN_BINS, dtype=jnp.bool_)
        opportunity_by_bin = opportunity[:, :, None] & in_bin[:, None, :]
        next_seen = seen + opportunity_by_bin.astype(jnp.int32)

        rng, reservoir_key, action_key = jax.random.split(rng, 3)
        draw = jax.random.uniform(reservoir_key, (games, 2, 1))
        replace = opportunity_by_bin & (
            draw < 1.0 / jnp.maximum(next_seen, 1)
        )
        sampled_states = _replace_slots(sampled_states, states, replace)
        sampled_memory_zero = _replace_slots(
            sampled_memory_zero, prior_memory_zero, replace
        )
        sampled_memory_one = _replace_slots(
            sampled_memory_one, prior_memory_one, replace
        )
        seen = next_seen
        found |= opportunity_by_bin

        action_keys = jax.random.split(action_key, 2 * games)
        action_indices = jax.vmap(jax.random.categorical)(action_keys, logits)
        actions = jax.vmap(decode_action)(action_indices.astype(jnp.int32))
        actions = jnp.stack([actions[:games], actions[games:]], axis=1)
        timesteps, _ = jax.vmap(
            lambda state, action: environment.step(state, action, pool)
        )(states, actions)
        active = ~finished
        states = jax.tree.map(
            lambda old, new: jnp.where(
                active.reshape(active.shape + (1,) * (old.ndim - 1)), new, old
            ),
            states,
            timesteps.last_state,
        )
        memory_zero = jax.tree.map(lambda value: value[:games], memories)
        memory_one = jax.tree.map(lambda value: value[games:], memories)
        finished |= timesteps.terminated | timesteps.truncated
        return (
            states,
            rng,
            memory_zero,
            memory_one,
            sampled_states,
            sampled_memory_zero,
            sampled_memory_one,
            seen,
            found,
            finished,
        ), None

    initial = (
        states,
        key,
        memory_zero,
        memory_one,
        sampled_states,
        sampled_memory_zero,
        sampled_memory_one,
        seen,
        found,
        finished,
    )
    final, _ = jax.lax.scan(step, initial, None, length=config.truncation)
    return {
        "states": final[4],
        "memory_zero": final[5],
        "memory_one": final[6],
        "opportunity_steps": final[7],
        "found": final[8],
    }


@dataclass
class CastleCounterfactualBuffer:
    """Bounded host replay with atomic, versioned refresh shards."""

    capacity: int
    max_age: int
    repetitions: int
    run_dir: Path
    data: dict[str, np.ndarray] | None = None

    @property
    def size(self) -> int:
        return 0 if self.data is None else int(self.data["build_action"].shape[0])

    def add(self, refresh: dict[str, np.ndarray], iteration: int) -> Path:
        self._validate_refresh(refresh)
        refresh = {name: np.asarray(value) for name, value in refresh.items()}
        self._merge(refresh, iteration)
        return self._write_shard(refresh, iteration)

    def _merge(self, refresh: dict[str, np.ndarray], iteration: int) -> None:
        if self.data is None:
            combined = refresh
        else:
            combined = {
                name: np.concatenate([self.data[name], refresh[name]], axis=0)
                for name in refresh
            }
        recent = combined["generator_iteration"] >= iteration - self.max_age
        combined = {name: value[recent] for name, value in combined.items()}
        if len(combined["build_action"]) > self.capacity:
            keep = np.argsort(combined["generator_iteration"], kind="stable")[-self.capacity :]
            combined = {name: value[keep] for name, value in combined.items()}
        self.data = combined

    def _validate_refresh(self, refresh: dict[str, np.ndarray]) -> None:
        required = set(ACTOR_FIELDS) | set(SUCCESSOR_FIELDS)
        missing = sorted(required - set(refresh))
        if missing:
            raise ValueError(f"Counterfactual refresh is missing fields: {missing}")
        lengths = {np.asarray(refresh[name]).shape[0] for name in required}
        if len(lengths) != 1:
            raise ValueError(f"Counterfactual field lengths disagree: {lengths}")
        if np.asarray(refresh["successor_target"]).shape[1:3] != (
            self.repetitions,
            2,
        ):
            raise ValueError("Counterfactual successor branch/repetition shape mismatch")
        masks = np.asarray(refresh["pre_mask"])
        builds = np.asarray(refresh["build_action"])
        controls = np.asarray(refresh["control_action"])
        rows = np.arange(len(builds))
        if not np.all(masks[rows, builds]):
            raise ValueError("Counterfactual buffer contains an illegal build")
        if not np.all(masks[rows, controls]):
            raise ValueError("Counterfactual buffer contains an illegal control")
        if not np.all((builds >= BUILD_START) & (builds < PASS_INDEX)):
            raise ValueError("Counterfactual build action has the wrong action kind")
        if np.any((controls >= BUILD_START) & (controls < PASS_INDEX)):
            raise ValueError("Counterfactual control action is a build")

    def _write_shard(self, refresh: dict[str, np.ndarray], iteration: int) -> Path:
        shard_dir = self.run_dir / "counterfactual_buffer"
        shard_dir.mkdir(parents=True, exist_ok=True)
        destination = shard_dir / f"refresh_{iteration:06d}.npz"
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                schema=np.asarray(SCHEMA_NAME),
                schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
                refresh_iteration=np.asarray(iteration, dtype=np.int32),
                **refresh,
            )
        os.replace(temporary, destination)
        digest = _sha256_file(destination)
        manifest = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "latest_refresh_iteration": iteration,
            "buffer_size": self.size,
            "refresh_examples": int(len(refresh["build_action"])),
            "latest_shard": str(destination),
            "latest_shard_sha256": digest,
        }
        manifest_path = shard_dir / "manifest.json"
        manifest_tmp = manifest_path.with_name(".manifest.json.tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        os.replace(manifest_tmp, manifest_path)
        return destination

    def load_completed_shards(self, through_iteration: int) -> None:
        shard_dir = self.run_dir / "counterfactual_buffer"
        for path in sorted(shard_dir.glob("refresh_*.npz")):
            iteration = int(path.stem.rsplit("_", 1)[1])
            if iteration > through_iteration:
                continue
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["schema"]) != SCHEMA_NAME:
                    raise ValueError(f"Incompatible counterfactual shard {path}")
                if int(archive["schema_version"]) != SCHEMA_VERSION:
                    raise ValueError(f"Unsupported counterfactual schema in {path}")
                refresh = {}
                for name in set(ACTOR_FIELDS) | set(SUCCESSOR_FIELDS):
                    value = archive[name]
                    # NumPy serializes ml_dtypes.bfloat16 as an opaque two-byte
                    # dtype in NPZ files. Restore the semantic dtype before JAX
                    # stages replay batches after a checkpoint continuation.
                    if value.dtype.kind == "V" and value.dtype.itemsize == 2:
                        value = value.view(jnp.bfloat16)
                    refresh[name] = value
            self._validate_refresh(refresh)
            self._merge(refresh, iteration)

    def _sample_partition(
        self,
        rng: np.random.Generator,
        candidates: np.ndarray,
        count: int,
        latest_refresh: int,
        recent_fraction: float,
        fallback: np.ndarray,
        avoid_games: set[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.unique(candidates)
        fallback = np.unique(fallback)
        recent = candidates[
            self.data["generator_iteration"][candidates]
            >= latest_refresh - 20
        ]
        old = candidates[
            self.data["generator_iteration"][candidates]
            < latest_refresh - 20
        ]
        recent_count = int(round(count * recent_fraction))
        old_count = count - recent_count
        selected: list[int] = []
        probabilities: list[float] = []
        selected_games = set() if avoid_games is None else set(avoid_games)

        def draw(pool: np.ndarray, amount: int, mass: float) -> None:
            if amount <= 0 or len(pool) == 0:
                return
            ordered = rng.permutation(pool)
            distinct = []
            remainder = []
            for index in ordered:
                game = int(self.data["source_game_id"][index])
                if game not in selected_games:
                    distinct.append(int(index))
                    selected_games.add(game)
                else:
                    remainder.append(int(index))
            chosen = (distinct + remainder)[:amount]
            if len(chosen) < amount:
                chosen.extend(
                    rng.choice(pool, size=amount - len(chosen), replace=True).tolist()
                )
            selected.extend(chosen)
            probabilities.extend([mass / len(pool)] * amount)

        draw(recent, recent_count, max(recent_fraction, 1e-12))
        draw(old, old_count, max(1.0 - recent_fraction, 1e-12))
        shortage = count - len(selected)
        if shortage:
            pool = np.setdiff1d(fallback, np.asarray(selected), assume_unique=False)
            if len(pool) == 0:
                pool = fallback
            draw(pool, shortage, 1.0)
        return np.asarray(selected, dtype=np.int32), np.asarray(probabilities, dtype=np.float32)

    @staticmethod
    def _cache_selected(
        fields: dict[str, np.ndarray], selected: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Build per-device unique caches plus scan-step cache indices."""
        devices = selected.shape[0]
        unique_by_device = [np.unique(selected[d]) for d in range(devices)]
        max_unique = max(len(value) for value in unique_by_device)
        cache_size = max(1, 1 << int(math.ceil(math.log2(max_unique))))
        cache = {}
        cache_indices = np.empty_like(selected, dtype=np.int32)
        for name, values in fields.items():
            device_values = []
            for device, unique in enumerate(unique_by_device):
                local = values[unique]
                padding = cache_size - len(unique)
                if padding:
                    local = np.concatenate(
                        [local, np.zeros((padding, *local.shape[1:]), dtype=local.dtype)]
                    )
                device_values.append(local)
                lookup = {int(source): index for index, source in enumerate(unique)}
                cache_indices[device] = np.vectorize(lookup.__getitem__)(
                    selected[device]
                )
            cache[name] = np.stack(device_values)
        return cache, cache_indices

    def sample_epoch(
        self,
        *,
        current_iteration: int,
        device_count: int,
        minibatches: int,
        seed: int,
        recent_fraction: float,
        actor_uniform_per_device: int,
        actor_positive_per_device: int,
        actor_negative_per_device: int,
        successor_per_device: int,
    ) -> dict[str, object]:
        if self.data is None or self.size == 0:
            raise ValueError("Cannot sample an empty counterfactual buffer")
        keep = (
            self.data["generator_iteration"]
            >= current_iteration - self.max_age
        )
        self.data = {name: value[keep] for name, value in self.data.items()}
        if self.size == 0:
            raise ValueError("All counterfactual examples expired before refresh")
        rng = np.random.default_rng(seed)
        all_indices = np.arange(self.size, dtype=np.int32)
        positive = all_indices[self.data["delta_shrunk"] > 0]
        negative = all_indices[self.data["delta_shrunk"] < 0]
        latest = int(self.data["generator_iteration"].max())
        actor_size = (
            actor_uniform_per_device
            + actor_positive_per_device
            + actor_negative_per_device
        )
        actor_selected = np.empty(
            (device_count, minibatches, actor_size), dtype=np.int32
        )
        actor_probability = np.empty_like(actor_selected, dtype=np.float32)
        for step in range(minibatches):
            for device in range(device_count):
                offset = 0
                used = np.empty((0,), dtype=np.int32)
                for candidates, count in (
                    (all_indices, actor_uniform_per_device),
                    (positive, actor_positive_per_device),
                    (negative, actor_negative_per_device),
                ):
                    available_candidates = np.setdiff1d(candidates, used)
                    available_fallback = np.setdiff1d(all_indices, used)
                    chosen, probability = self._sample_partition(
                        rng,
                        available_candidates,
                        count,
                        latest,
                        recent_fraction,
                        available_fallback,
                        {
                            int(self.data["source_game_id"][index])
                            for index in used
                        },
                    )
                    actor_selected[device, step, offset : offset + count] = chosen
                    actor_probability[device, step, offset : offset + count] = probability
                    used = np.concatenate([used, chosen])
                    offset += count
                permutation = rng.permutation(actor_size)
                actor_selected[device, step] = actor_selected[device, step, permutation]
                actor_probability[device, step] = actor_probability[device, step, permutation]

        # Normalize and conservatively clip inverse-probability corrections per
        # global minibatch. The raw component probability remains in the buffer
        # and metrics for auditability.
        inverse = 1.0 / np.maximum(actor_probability, 1e-12)
        inverse /= inverse.mean(axis=(0, 2), keepdims=True)
        inverse = np.clip(inverse, 0.25, 4.0).astype(np.float32)

        actor_fields = {name: self.data[name] for name in ACTOR_TRAIN_FIELDS}
        actor_cache, actor_cache_indices = self._cache_selected(
            actor_fields, actor_selected
        )

        pair_count = self.size * self.repetitions
        pair_ids = np.arange(pair_count, dtype=np.int32)
        successor_selected = np.empty(
            (device_count, minibatches, successor_per_device), dtype=np.int32
        )
        pair_ids_per_step = device_count * successor_per_device // 2
        if pair_ids_per_step * 2 != device_count * successor_per_device:
            raise ValueError("Global successor batch must contain complete branch pairs")
        for step in range(minibatches):
            chosen_pairs = rng.choice(
                pair_ids,
                size=pair_ids_per_step,
                replace=len(pair_ids) < pair_ids_per_step,
            )
            flattened = np.stack(
                [2 * chosen_pairs + BRANCH_BUILD, 2 * chosen_pairs + BRANCH_CONTROL],
                axis=-1,
            ).reshape(-1)
            rng.shuffle(flattened)
            successor_selected[:, step] = flattened.reshape(
                device_count, successor_per_device
            )

        successor_fields = {
            name: self.data[name].reshape(
                self.size * self.repetitions * 2,
                *self.data[name].shape[3:],
            )
            for name in SUCCESSOR_TRAIN_FIELDS
        }
        successor_cache, successor_cache_indices = self._cache_selected(
            successor_fields, successor_selected
        )
        return {
            "actor_cache": actor_cache,
            "actor_indices": actor_cache_indices,
            "actor_inverse_probability": inverse,
            "actor_raw_probability": actor_probability,
            "successor_cache": successor_cache,
            "successor_indices": successor_cache_indices,
            "positive_examples": int(len(positive)),
            "negative_examples": int(len(negative)),
            "uncertain_examples": int(self.size - len(positive) - len(negative)),
            "latest_generator_iteration": latest,
        }


def _select_by_seat(zero, one, seats):
    return jax.tree.map(
        lambda left, right: jnp.where(
            seats.reshape(seats.shape + (1,) * (left.ndim - seats.ndim)),
            right,
            left,
        ),
        zero,
        one,
    )


@eqx.filter_jit
def describe_sources(network, states, memory_zero, memory_one, seats, config, environment):
    """Build observable source features and neural inputs for host selection."""
    count = states.armies.shape[0]
    obs_zero = jax.vmap(lambda state: get_observation(state, 0))(states)
    obs_one = jax.vmap(lambda state: get_observation(state, 1))(states)
    observations = jax.tree.map(
        lambda zero, one: jnp.concatenate([zero, one]), obs_zero, obs_one
    )
    memories = jax.tree.map(
        lambda zero, one: jnp.concatenate([zero, one]), memory_zero, memory_one
    )
    board_masks = jnp.concatenate([states.board_mask, states.board_mask])
    augmented, updated_memories = jax.vmap(
        lambda observation, memory, board_mask: augment_observation(
            observation,
            memory,
            board_mask,
            config.observation_schema,
            environment.deathtouch_turn or 800,
        )
    )(observations, memories, board_masks)
    masks = jax.vmap(legal_action_mask)(observations, board_masks)
    histories = temporal_input(updated_memories)
    logits = jax.vmap(network.forward)(augmented, histories, masks)[0]
    actor_augmented = jnp.where(seats[:, None, None, None] == 0, augmented[:count], augmented[count:])
    actor_histories = jnp.where(seats[:, None, None] == 0, histories[:count], histories[count:])
    actor_masks = jnp.where(seats[:, None] == 0, masks[:count], masks[count:])
    actor_logits = jnp.where(seats[:, None] == 0, logits[:count], logits[count:])
    actor_observations = _select_by_seat(obs_zero, obs_one, seats)
    costs = jax.vmap(build_cost_grid_from_observation)(actor_observations)
    post_garrison = actor_observations.armies - costs
    legal_builds = actor_masks[:, BUILD_START:PASS_INDEX].reshape(-1, 21, 21)
    best_build_logits = jnp.max(actor_logits[:, BUILD_START:PASS_INDEX], axis=-1)
    nonbuild_logits = actor_logits.at[:, BUILD_START:PASS_INDEX].set(-1e9)
    control_actions = jnp.argmax(nonbuild_logits, axis=-1).astype(jnp.int32)
    best_control_logits = jnp.take_along_axis(
        actor_logits, control_actions[:, None], axis=1
    )[:, 0]

    rows = jnp.arange(21)[None, :, None]
    cols = jnp.arange(21)[None, None, :]
    own_generals = actor_observations.generals & actor_observations.owned_cells
    general_flat = jnp.argmax(own_generals.reshape(count, -1), axis=-1)
    general_row = general_flat // 21
    general_col = general_flat % 21
    general_distance = jnp.abs(rows - general_row[:, None, None]) + jnp.abs(
        cols - general_col[:, None, None]
    )
    visible_enemy = actor_observations.opponent_cells
    safest_site = jnp.argmax(
        jnp.where(legal_builds, post_garrison, -1e9).reshape(count, -1), axis=-1
    )
    site_row, site_col = safest_site // 21, safest_site % 21
    site_cost = costs[jnp.arange(count), site_row, site_col]
    enemy_distance = jax.vmap(
        lambda enemy, row, col: jnp.min(
            jnp.where(
                enemy,
                jnp.abs(rows[0] - row) + jnp.abs(cols[0] - col),
                999,
            )
        )
    )(visible_enemy, site_row, site_col)
    return {
        "pre_observation": actor_augmented.astype(jnp.bfloat16),
        "pre_history": actor_histories.astype(jnp.bfloat16),
        "pre_mask": actor_masks,
        "actor_logits": actor_logits,
        "control_action": control_actions,
        "legal_builds": legal_builds,
        "post_garrison": post_garrison,
        "site_cost": site_cost,
        "general_distance": general_distance,
        "enemy_distance": enemy_distance,
        "turn": states.time,
        "army_margin": actor_observations.owned_army_count
        - actor_observations.opponent_army_count,
        "land_margin": actor_observations.owned_land_count
        - actor_observations.opponent_land_count,
        "best_build_margin": best_build_logits - best_control_logits,
    }


def _balanced_feature_diverse(features: dict[str, np.ndarray], candidates: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or len(candidates) == 0:
        return np.empty((0,), dtype=np.int32)
    names = ("turn", "army_margin", "land_margin", "site_cost")
    binned = []
    for name in names:
        values = np.asarray(features[name])[candidates]
        quantiles = np.unique(np.quantile(values, [0.25, 0.5, 0.75]))
        binned.append(np.digitize(values, quantiles))
    garrison = np.max(
        np.where(
            features["legal_builds"][candidates],
            features["post_garrison"][candidates],
            -1e9,
        ),
        axis=(1, 2),
    )
    binned.append(np.digitize(garrison, [0, 10, 25]))
    keys = list(zip(*(value.tolist() for value in binned), strict=True))
    buckets: dict[tuple[int, ...], list[int]] = {}
    for candidate, key in zip(candidates.tolist(), keys, strict=True):
        buckets.setdefault(key, []).append(candidate)
    selected = []
    while len(selected) < count and buckets:
        for key in sorted(list(buckets)):
            selected.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
            if len(selected) == count:
                break
    return np.asarray(selected, dtype=np.int32)


def select_source_quotas(
    features: dict[str, np.ndarray], config, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select exact 128/64/64 source quotas with deterministic backfill."""
    rng = np.random.default_rng(seed)
    total = len(features["turn"])
    all_indices = np.arange(total, dtype=np.int32)
    chosen: list[int] = []
    chosen_games: set[int] = set()
    quotas: list[int] = []
    probabilities: list[float] = []

    def add(pool: np.ndarray, count: int, quota: int, deterministic=False):
        available = np.setdiff1d(pool, np.asarray(chosen, dtype=np.int32))
        short = min(count, len(available))
        ordered = available if deterministic else rng.permutation(available)
        distinct = [
            int(index)
            for index in ordered
            if int(features["source_game_id"][index]) not in chosen_games
        ]
        remainder = [int(index) for index in ordered if int(index) not in distinct]
        selected = np.asarray((distinct + remainder)[:short], dtype=np.int32)
        probability = 1.0 if deterministic else short / max(len(available), 1)
        chosen.extend(np.asarray(selected, dtype=np.int32).tolist())
        chosen_games.update(
            int(features["source_game_id"][index]) for index in selected
        )
        quotas.extend([quota] * short)
        probabilities.extend([probability] * short)

    add(all_indices, config.counterfactual_uniform_source_states, 0)
    max_garrison = np.max(
        np.where(features["legal_builds"], features["post_garrison"], -1e9),
        axis=(1, 2),
    )
    min_enemy = features["enemy_distance"]
    promising = all_indices[
        (features["turn"] <= 700)
        & ((features["army_margin"] < 0) | (features["land_margin"] < 0))
        & (max_garrison >= 10)
        & (min_enemy > 3)
    ]
    add(promising, config.counterfactual_promising_source_states, 1)

    remaining = np.setdiff1d(all_indices, np.asarray(chosen, dtype=np.int32))
    margin_order = remaining[
        np.argsort(features["best_build_margin"][remaining], kind="stable")[::-1]
    ]
    add(margin_order, config.counterfactual_hard_source_states // 2, 2, deterministic=True)
    remaining = np.setdiff1d(all_indices, np.asarray(chosen, dtype=np.int32))
    diverse = _balanced_feature_diverse(
        features,
        remaining,
        config.counterfactual_hard_source_states
        - config.counterfactual_hard_source_states // 2,
    )
    add(diverse, len(diverse), 2, deterministic=True)

    shortfall = config.counterfactual_source_states - len(chosen)
    if shortfall:
        add(all_indices, shortfall, 0)
    if len(chosen) != config.counterfactual_source_states:
        raise RuntimeError(
            f"Only {len(chosen)} distinct grouped build opportunities; "
            f"need {config.counterfactual_source_states}"
        )
    return (
        np.asarray(chosen, dtype=np.int32),
        np.asarray(quotas, dtype=np.int8),
        np.asarray(probabilities, dtype=np.float32),
    )


def propose_build_candidates(
    features: dict[str, np.ndarray], source_ids: np.ndarray, iteration: int
) -> dict[str, np.ndarray]:
    """Emit highest-logit plus stable rotating non-duplicate build proposals."""
    actions = []
    source_rows = []
    methods = []
    probabilities = []
    for row in range(len(source_ids)):
        legal = np.flatnonzero(features["legal_builds"][row].reshape(-1))
        if len(legal) == 0:
            raise ValueError("Selected source has no legal build")
        build_logits = features["actor_logits"][row, BUILD_START:PASS_INDEX]
        first = int(legal[np.argmax(build_logits[legal])])
        proposed = [first]
        proposed_methods = [0]
        proposed_probabilities = [1.0]
        digest = hashlib.blake2b(
            f"{int(source_ids[row])}:{iteration}".encode(), digest_size=8
        ).digest()
        rotation = int.from_bytes(digest, "little") % 3
        for method in ((rotation + offset) % 3 for offset in range(3)):
            if method == 0:
                scores = features["post_garrison"][row].reshape(-1)
                candidate = int(legal[np.argmax(scores[legal])])
                probability = 1.0
            elif method == 1:
                garrison = features["post_garrison"][row].reshape(-1)[legal]
                weights = np.where(garrison >= 10, np.maximum(garrison + 1, 1), 1.0)
                weights = weights / weights.sum()
                local_rng = np.random.default_rng(
                    int.from_bytes(digest, "little") ^ 0x7919
                )
                local = int(local_rng.choice(len(legal), p=weights))
                candidate = int(legal[local])
                probability = float(weights[local])
            else:
                distance = features["general_distance"][row].reshape(-1)
                candidate = int(legal[np.argmin(distance[legal])])
                probability = 1.0
            if candidate not in proposed:
                proposed.append(candidate)
                proposed_methods.append(method + 1)
                proposed_probabilities.append(probability)
                break
        for candidate, method, probability in zip(
            proposed, proposed_methods, proposed_probabilities, strict=True
        ):
            source_rows.append(row)
            actions.append(BUILD_START + candidate)
            methods.append(method)
            probabilities.append(probability)
    return {
        "source_row": np.asarray(source_rows, dtype=np.int32),
        "build_action": np.asarray(actions, dtype=np.int32),
        "proposal_method": np.asarray(methods, dtype=np.int8),
        "proposal_probability": np.asarray(probabilities, dtype=np.float32),
    }


def _expand_pairs(value, repetitions: int):
    count = value.shape[0]
    return jnp.broadcast_to(
        value[:, None, None], (count, repetitions, 2, *value.shape[1:])
    ).reshape(count * repetitions * 2, *value.shape[1:])


@eqx.filter_jit
def evaluate_candidate_pairs(
    network,
    source_states,
    source_memory_zero,
    source_memory_one,
    actor_seats,
    build_indices,
    control_indices,
    pool,
    key,
    config,
    environment,
    *,
    repetitions: int,
    rollout_steps: int,
):
    """Force build/control actions and retain distinct successor inputs."""
    count = source_states.armies.shape[0]
    obs_zero = jax.vmap(lambda state: get_observation(state, 0))(source_states)
    obs_one = jax.vmap(lambda state: get_observation(state, 1))(source_states)
    observations = jax.tree.map(
        lambda zero, one: jnp.concatenate([zero, one]), obs_zero, obs_one
    )
    memories = jax.tree.map(
        lambda zero, one: jnp.concatenate([zero, one]),
        source_memory_zero,
        source_memory_one,
    )
    board_masks = jnp.concatenate([source_states.board_mask, source_states.board_mask])
    augmented, updated_memories = jax.vmap(
        lambda observation, memory, board_mask: augment_observation(
            observation,
            memory,
            board_mask,
            config.observation_schema,
            environment.deathtouch_turn or 800,
        )
    )(observations, memories, board_masks)
    masks = jax.vmap(legal_action_mask)(observations, board_masks)
    histories = temporal_input(updated_memories)
    logits = jax.vmap(network.forward)(augmented, histories, masks)[0]
    opponent_logits = jnp.where(
        actor_seats[:, None] == 0, logits[count:], logits[:count]
    )

    states = jax.tree.map(
        lambda value: _expand_pairs(value, repetitions), source_states
    )
    memory_zero = jax.tree.map(
        lambda value: _expand_pairs(value, repetitions),
        jax.tree.map(lambda value: value[:count], updated_memories),
    )
    memory_one = jax.tree.map(
        lambda value: _expand_pairs(value, repetitions),
        jax.tree.map(lambda value: value[count:], updated_memories),
    )
    actor_flat = _expand_pairs(actor_seats, repetitions)
    build_flat = _expand_pairs(build_indices, repetitions)
    control_flat = _expand_pairs(control_indices, repetitions)
    branch = jnp.broadcast_to(
        jnp.arange(2, dtype=jnp.int32)[None, None, :],
        (count, repetitions, 2),
    ).reshape(-1)
    actor_index = jnp.where(branch == BRANCH_BUILD, build_flat, control_flat)

    key, intervention_key = jax.random.split(key)
    opponent_keys = jax.random.split(intervention_key, count * repetitions)
    repeated_opponent_logits = jnp.broadcast_to(
        opponent_logits[:, None],
        (count, repetitions, opponent_logits.shape[-1]),
    ).reshape(count * repetitions, -1)
    opponent_index = jax.vmap(jax.random.categorical)(
        opponent_keys, repeated_opponent_logits
    )
    opponent_index = jnp.repeat(opponent_index, 2)
    actor_actions = jax.vmap(decode_action)(actor_index.astype(jnp.int32))
    opponent_actions = jax.vmap(decode_action)(opponent_index.astype(jnp.int32))
    action_zero = jnp.where(
        actor_flat[:, None] == 0, actor_actions, opponent_actions
    )
    action_one = jnp.where(
        actor_flat[:, None] == 1, actor_actions, opponent_actions
    )
    joint_actions = jnp.stack([action_zero, action_one], axis=1)
    intervention, _ = jax.vmap(
        lambda state, action: environment.step(state, action, pool)
    )(states, joint_actions)
    states = intervention.last_state
    done = intervention.terminated | intervention.truncated
    build_offset = jnp.clip(build_flat - BUILD_START, 0, CELL_COUNT - 1)
    build_row, build_col = build_offset // 21, build_offset % 21
    flat_index = jnp.arange(states.armies.shape[0])
    # Ownership may change later in the same simultaneous tick, but the castle
    # bit proves the legal build itself was applied before movement resolution.
    constructed = states.castles[flat_index, build_row, build_col]

    # Capture V-training inputs before either branch samples its next action.
    post_obs_zero = jax.vmap(lambda state: get_observation(state, 0))(states)
    post_obs_one = jax.vmap(lambda state: get_observation(state, 1))(states)
    post_observations = jax.tree.map(
        lambda zero, one: jnp.concatenate([zero, one]), post_obs_zero, post_obs_one
    )
    post_memories = jax.tree.map(
        lambda zero, one: jnp.concatenate([zero, one]), memory_zero, memory_one
    )
    post_board_masks = jnp.concatenate([states.board_mask, states.board_mask])
    post_augmented, post_updated = jax.vmap(
        lambda observation, memory, board_mask: augment_observation(
            observation,
            memory,
            board_mask,
            config.observation_schema,
            environment.deathtouch_turn or 800,
        )
    )(post_observations, post_memories, post_board_masks)
    post_masks = jax.vmap(legal_action_mask)(post_observations, post_board_masks)
    post_histories = temporal_input(post_updated)
    games = states.armies.shape[0]
    successor_observation = jnp.where(
        actor_flat[:, None, None, None] == 0,
        post_augmented[:games],
        post_augmented[games:],
    )
    successor_history = jnp.where(
        actor_flat[:, None, None] == 0,
        post_histories[:games],
        post_histories[games:],
    )
    successor_mask = jnp.where(
        actor_flat[:, None] == 0, post_masks[:games], post_masks[games:]
    )

    winner = intervention.info.winner
    actor_won = winner == actor_flat
    actor_lost = (winner >= 0) & ~actor_won
    target = jnp.where(actor_won, 1.0, jnp.where(actor_lost, -1.0, 0.0))
    target = jnp.where(done, target, 0.0)

    def continuation_step(carry, _):
        states, rng, memory_zero, memory_one, done, target = carry
        games = states.armies.shape[0]
        obs_zero = jax.vmap(lambda state: get_observation(state, 0))(states)
        obs_one = jax.vmap(lambda state: get_observation(state, 1))(states)
        observations = jax.tree.map(
            lambda zero, one: jnp.concatenate([zero, one]), obs_zero, obs_one
        )
        memories = jax.tree.map(
            lambda zero, one: jnp.concatenate([zero, one]),
            memory_zero,
            memory_one,
        )
        board_masks = jnp.concatenate([states.board_mask, states.board_mask])
        augmented, memories = jax.vmap(
            lambda observation, memory, board_mask: augment_observation(
                observation,
                memory,
                board_mask,
                config.observation_schema,
                environment.deathtouch_turn or 800,
            )
        )(observations, memories, board_masks)
        masks = jax.vmap(legal_action_mask)(observations, board_masks)
        histories = temporal_input(memories)
        logits = jax.vmap(network.forward)(augmented, histories, masks)[0]

        rng, action_key = jax.random.split(rng)
        paired_games = count * repetitions
        common_keys = jax.random.split(action_key, paired_games * 2).reshape(
            paired_games, 2, 2
        )
        keys_zero = jnp.repeat(common_keys[:, 0], 2, axis=0)
        keys_one = jnp.repeat(common_keys[:, 1], 2, axis=0)
        index_zero = jax.vmap(jax.random.categorical)(keys_zero, logits[:games])
        index_one = jax.vmap(jax.random.categorical)(keys_one, logits[games:])
        actions = jnp.stack(
            [
                jax.vmap(decode_action)(index_zero.astype(jnp.int32)),
                jax.vmap(decode_action)(index_one.astype(jnp.int32)),
            ],
            axis=1,
        )
        active = ~done
        timestep, _ = jax.vmap(
            lambda state, action: environment.step(state, action, pool)
        )(states, actions)
        states = jax.tree.map(
            lambda old, new: jnp.where(
                active.reshape(active.shape + (1,) * (old.ndim - 1)), new, old
            ),
            states,
            timestep.last_state,
        )
        memory_zero = jax.tree.map(lambda value: value[:games], memories)
        memory_one = jax.tree.map(lambda value: value[games:], memories)
        newly_done = active & (timestep.terminated | timestep.truncated)
        winner = timestep.info.winner
        actor_won = winner == actor_flat
        actor_lost = (winner >= 0) & ~actor_won
        new_target = jnp.where(
            actor_won, 1.0, jnp.where(actor_lost, -1.0, 0.0)
        )
        target = jnp.where(newly_done, new_target, target)
        done |= newly_done
        return (states, rng, memory_zero, memory_one, done, target), None

    final, _ = jax.lax.scan(
        continuation_step,
        (states, key, memory_zero, memory_one, done, target),
        None,
        length=rollout_steps - 1,
    )
    return {
        "successor_observation": successor_observation.reshape(
            count, repetitions, 2, *successor_observation.shape[1:]
        ).astype(jnp.bfloat16),
        "successor_history": successor_history.reshape(
            count, repetitions, 2, *successor_history.shape[1:]
        ).astype(jnp.bfloat16),
        "successor_mask": successor_mask.reshape(
            count, repetitions, 2, *successor_mask.shape[1:]
        ),
        "successor_target": final[5].reshape(count, repetitions, 2),
        "successor_finished": final[4].reshape(count, repetitions, 2),
        "opponent_action": opponent_index.reshape(count, repetitions, 2),
        "constructed": constructed.reshape(count, repetitions, 2),
    }


def construct_refresh(
    *,
    source_features: dict[str, np.ndarray],
    source_ids: np.ndarray,
    source_seats: np.ndarray,
    source_quotas: np.ndarray,
    source_probabilities: np.ndarray,
    proposals: dict[str, np.ndarray],
    paired: dict[str, np.ndarray],
    iteration: int,
    uncertainty_kappa: float,
) -> dict[str, np.ndarray]:
    rows = proposals["source_row"]
    outcomes = np.asarray(paired["successor_target"], dtype=np.float32)
    delta_repetitions = outcomes[:, :, BRANCH_BUILD] - outcomes[:, :, BRANCH_CONTROL]
    delta_mean = delta_repetitions.mean(axis=1)
    if delta_repetitions.shape[1] > 1:
        delta_se = delta_repetitions.std(axis=1, ddof=1) / math.sqrt(
            delta_repetitions.shape[1]
        )
    else:
        delta_se = np.zeros_like(delta_mean)
    delta_shrunk = np.sign(delta_mean) * np.maximum(
        np.abs(delta_mean) - uncertainty_kappa * delta_se, 0.0
    )
    candidate_ordinal = np.zeros(len(rows), dtype=np.uint64)
    seen_per_source: dict[int, int] = {}
    for index, row in enumerate(rows.tolist()):
        candidate_ordinal[index] = seen_per_source.get(row, 0)
        seen_per_source[row] = int(candidate_ordinal[index]) + 1
    return {
        "pre_observation": np.asarray(source_features["pre_observation"])[rows],
        "pre_history": np.asarray(source_features["pre_history"])[rows],
        "pre_mask": np.asarray(source_features["pre_mask"])[rows],
        "build_action": proposals["build_action"],
        "control_action": np.asarray(source_features["control_action"])[rows],
        "delta_mean": delta_mean.astype(np.float32),
        "delta_se": delta_se.astype(np.float32),
        "delta_shrunk": delta_shrunk.astype(np.float32),
        "source_id": source_ids[rows].astype(np.uint64) * np.uint64(2)
        + candidate_ordinal,
        "source_game_id": np.asarray(source_features["source_game_id"])[rows].astype(
            np.uint64
        ),
        "source_turn": np.asarray(source_features["turn"])[rows].astype(np.int32),
        "source_seat": source_seats[rows].astype(np.int8),
        "source_quota": source_quotas[rows].astype(np.int8),
        "source_selection_probability": source_probabilities[rows].astype(np.float32),
        "proposal_method": proposals["proposal_method"],
        "proposal_probability": proposals["proposal_probability"],
        "generator_iteration": np.full(len(rows), iteration, dtype=np.int32),
        "successor_observation": np.asarray(paired["successor_observation"]),
        "successor_history": np.asarray(paired["successor_history"]),
        "successor_mask": np.asarray(paired["successor_mask"]),
        "successor_target": outcomes,
        "successor_finished": np.asarray(paired["successor_finished"]),
    }


def _pad_tree_to_devices(tree, device_count: int):
    count = jax.tree.leaves(tree)[0].shape[0]
    padded = int(math.ceil(count / device_count) * device_count)
    if padded == count:
        return tree, count
    return jax.tree.map(
        lambda value: jnp.concatenate(
            [value, jnp.repeat(value[:1], padded - count, axis=0)], axis=0
        ),
        tree,
    ), count


def _shard_tree(tree, device_count: int):
    count = jax.tree.leaves(tree)[0].shape[0]
    return jax.tree.map(
        lambda value: value.reshape(
            device_count, count // device_count, *value.shape[1:]
        ),
        tree,
    )


def _flatten_device_tree(tree, original_count: int):
    return jax.tree.map(
        lambda value: np.asarray(jax.device_get(value)).reshape(
            (-1, *value.shape[2:])
        )[:original_count],
        tree,
    )


def _network_hash(network) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(eqx.filter(network, eqx.is_inexact_array)):
        value = np.asarray(jax.device_get(leaf))
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def generate_counterfactual_refresh(
    *,
    network,
    pool_replicated,
    key,
    config,
    environment,
    iteration: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Generate one full refresh sharded across all visible training devices."""
    device_count = jax.device_count()
    games = config.counterfactual_source_games_per_device
    if games * device_count > environment.pool_size:
        raise ValueError("Counterfactual source games exceed the environment pool")
    key, collect_key, pair_key = jax.random.split(key, 3)
    collect_keys = jax.random.split(collect_key, device_count)
    offsets = jnp.arange(device_count, dtype=jnp.int32) * games

    def collect_shard(shard_network, pool_shard, rng, offset):
        source_pool = jax.tree.map(
            lambda value: jax.lax.dynamic_slice_in_dim(value, offset, games, axis=0),
            pool_shard,
        )
        return collect_grouped_opportunities(
            shard_network, source_pool, rng, config, environment
        )

    collected = jax.pmap(collect_shard, in_axes=(None, 0, 0, 0))(
        network, pool_replicated, collect_keys, offsets
    )
    collected_host = jax.device_get(collected)
    del collected
    found = np.asarray(collected_host["found"]).reshape(-1)
    if found.sum() < config.counterfactual_source_states:
        raise RuntimeError(
            f"Only {int(found.sum())} grouped legal-build opportunities were found"
        )
    raw_states = jax.tree.map(
        lambda value: np.asarray(value).reshape((-1, *value.shape[4:]))[found],
        collected_host["states"],
    )
    raw_memory_zero = jax.tree.map(
        lambda value: np.asarray(value).reshape((-1, *value.shape[4:]))[found],
        collected_host["memory_zero"],
    )
    raw_memory_one = jax.tree.map(
        lambda value: np.asarray(value).reshape((-1, *value.shape[4:]))[found],
        collected_host["memory_one"],
    )
    shape = np.asarray(collected_host["found"]).shape
    seat_grid = np.broadcast_to(
        np.arange(2, dtype=np.int8)[None, None, :, None], shape
    ).reshape(-1)[found]
    device_grid = np.broadcast_to(
        np.arange(device_count, dtype=np.int32)[:, None, None, None], shape
    ).reshape(-1)[found]
    game_grid = np.broadcast_to(
        np.arange(games, dtype=np.int32)[None, :, None, None], shape
    ).reshape(-1)[found]
    bin_grid = np.broadcast_to(
        np.arange(TURN_BINS, dtype=np.int32)[None, None, None, :], shape
    ).reshape(-1)[found]
    source_ids = np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{iteration}:{d}:{g}:{s}:{b}".encode(), digest_size=8
                ).digest(),
                "little",
            )
            for d, g, s, b in zip(
                device_grid, game_grid, seat_grid, bin_grid, strict=True
            )
        ],
        dtype=np.uint64,
    )

    describe_inputs = {
        "states": jax.tree.map(jnp.asarray, raw_states),
        "memory_zero": jax.tree.map(jnp.asarray, raw_memory_zero),
        "memory_one": jax.tree.map(jnp.asarray, raw_memory_one),
        "seats": jnp.asarray(seat_grid),
    }
    describe_inputs, describe_count = _pad_tree_to_devices(
        describe_inputs, device_count
    )
    describe_inputs = _shard_tree(describe_inputs, device_count)

    def describe_shard(shard_network, payload):
        return describe_sources(
            shard_network,
            payload["states"],
            payload["memory_zero"],
            payload["memory_one"],
            payload["seats"],
            config,
            environment,
        )

    described = jax.pmap(describe_shard, in_axes=(None, 0))(
        network, describe_inputs
    )
    features = _flatten_device_tree(described, describe_count)
    del described, describe_inputs
    features["source_game_id"] = np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"game:{iteration}:{d}:{g}".encode(), digest_size=8
                ).digest(),
                "little",
            )
            for d, g in zip(device_grid, game_grid, strict=True)
        ],
        dtype=np.uint64,
    )
    selected, quotas, source_probability = select_source_quotas(
        features, config, seed=iteration ^ 0xCA571E
    )
    selected_states = jax.tree.map(lambda value: jnp.asarray(value[selected]), raw_states)
    selected_memory_zero = jax.tree.map(
        lambda value: jnp.asarray(value[selected]), raw_memory_zero
    )
    selected_memory_one = jax.tree.map(
        lambda value: jnp.asarray(value[selected]), raw_memory_one
    )
    selected_seats = seat_grid[selected]
    selected_ids = source_ids[selected]
    selected_features = {name: value[selected] for name, value in features.items()}
    proposals = propose_build_candidates(selected_features, selected_ids, iteration)
    rows = proposals["source_row"]
    candidate_inputs = {
        "states": jax.tree.map(lambda value: value[rows], selected_states),
        "memory_zero": jax.tree.map(lambda value: value[rows], selected_memory_zero),
        "memory_one": jax.tree.map(lambda value: value[rows], selected_memory_one),
        "seats": jnp.asarray(selected_seats[rows]),
        "build": jnp.asarray(proposals["build_action"]),
        "control": jnp.asarray(selected_features["control_action"][rows]),
    }
    candidate_inputs, candidate_count = _pad_tree_to_devices(
        candidate_inputs, device_count
    )
    candidate_inputs = _shard_tree(candidate_inputs, device_count)
    pair_keys = jax.random.split(pair_key, device_count)

    def pair_shard(shard_network, payload, pool_shard, rng):
        return evaluate_candidate_pairs(
            shard_network,
            payload["states"],
            payload["memory_zero"],
            payload["memory_one"],
            payload["seats"],
            payload["build"],
            payload["control"],
            pool_shard,
            rng,
            config,
            environment,
            repetitions=config.counterfactual_repetitions,
            rollout_steps=config.counterfactual_rollout_steps,
        )

    paired = jax.pmap(pair_shard, in_axes=(None, 0, 0, 0))(
        network, candidate_inputs, pool_replicated, pair_keys
    )
    paired = _flatten_device_tree(paired, candidate_count)
    if not np.all(paired["successor_finished"]):
        unfinished = int(np.size(paired["successor_finished"]) - paired["successor_finished"].sum())
        raise RuntimeError(f"Counterfactual refresh left {unfinished} branches unfinished")
    constructed = np.asarray(paired["constructed"])
    if not np.all(constructed[:, :, BRANCH_BUILD]):
        failures = int((~constructed[:, :, BRANCH_BUILD]).sum())
        raise RuntimeError(
            f"{failures} legal forced build actions did not construct castles"
        )
    refresh = construct_refresh(
        source_features=selected_features,
        source_ids=selected_ids,
        source_seats=selected_seats,
        source_quotas=quotas,
        source_probabilities=source_probability,
        proposals=proposals,
        paired=paired,
        iteration=iteration,
        uncertainty_kappa=config.counterfactual_uncertainty_kappa,
    )
    metrics = {
        "counterfactual/schema_version": SCHEMA_VERSION,
        "counterfactual/generator_iteration": iteration,
        "counterfactual/generator_hash": _network_hash(network),
        "counterfactual/grouped_opportunities": int(found.sum()),
        "counterfactual/source_states": int(len(selected)),
        "counterfactual/candidates": int(len(refresh["build_action"])),
        "counterfactual/paired_continuations": int(
            len(refresh["build_action"]) * config.counterfactual_repetitions
        ),
        "counterfactual/positive_labels": int((refresh["delta_shrunk"] > 0).sum()),
        "counterfactual/negative_labels": int((refresh["delta_shrunk"] < 0).sum()),
        "counterfactual/uncertain_labels": int((refresh["delta_shrunk"] == 0).sum()),
        "counterfactual/delta_mean": float(refresh["delta_mean"].mean()),
        "counterfactual/delta_se_mean": float(refresh["delta_se"].mean()),
    }
    return refresh, metrics

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from generals.training.actions import PASS_INDEX
from generals.training.config import TrainingConfig
from generals.training.counterfactual import (
    ACTOR_FIELDS,
    BUILD_START,
    CastleCounterfactualBuffer,
    SUCCESSOR_FIELDS,
    propose_build_candidates,
)
from generals.training.train import (
    _attach_zero_build_difference_head,
    _migrate_optimizer_state_with_zero_head,
    build_network,
)


def _small_config(counterfactual: bool) -> TrainingConfig:
    return TrainingConfig(
        observation_schema="competition_39",
        model_architecture="conv_transformer",
        depth=1,
        embed_dim=32,
        attention_heads=4,
        ff_factor=2,
        value_bins=16,
        use_bf16=False,
        conv_channels=12,
        conv_groups=3,
        counterfactual_castle_training=counterfactual,
        residual_build_kind_head=counterfactual,
    )


def test_zero_head_migration_preserves_actor_and_critic_exactly():
    old_config = _small_config(False)
    new_config = replace(
        old_config,
        counterfactual_castle_training=True,
        residual_build_kind_head=True,
    )
    key = jax.random.PRNGKey(1)
    old = build_network(old_config, key)
    initialized = build_network(new_config, key)
    migrated = _attach_zero_build_difference_head(old, initialized)
    observation = jax.random.normal(jax.random.PRNGKey(2), (39, 21, 21))
    history = jnp.zeros((2, 512))
    mask = jnp.ones((3970,), dtype=jnp.bool_)
    for expected, actual in zip(
        old.forward(observation, history, mask),
        migrated.forward(observation, history, mask),
        strict=True,
    ):
        np.testing.assert_array_equal(expected, actual)
    difference = migrated.build_difference(observation, history, mask)
    np.testing.assert_array_equal(difference, np.zeros((21, 21)))


def test_optimizer_migration_preserves_old_moments_and_zeros_new_head():
    old_config = _small_config(False)
    new_config = replace(
        old_config,
        counterfactual_castle_training=True,
        residual_build_kind_head=True,
    )
    old = build_network(old_config, jax.random.PRNGKey(3))
    new = build_network(new_config, jax.random.PRNGKey(3))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))
    old_state = optimizer.init(eqx.filter(old, eqx.is_inexact_array))
    new_state = optimizer.init(eqx.filter(new, eqx.is_inexact_array))
    migrated = _migrate_optimizer_state_with_zero_head(old_state, new_state)
    old_adam = old_state[1][0] if isinstance(old_state[1], tuple) else old_state[1]
    migrated_adam = migrated[1][0] if isinstance(migrated[1], tuple) else migrated[1]
    np.testing.assert_array_equal(
        migrated_adam.mu.transformer.value_token,
        old_adam.mu.transformer.value_token,
    )
    assert np.all(np.asarray(migrated_adam.mu.build_difference_head.weight) == 0)
    assert np.all(np.asarray(migrated_adam.nu.build_difference_head.weight) == 0)
    assert np.all(np.asarray(migrated_adam.mu.build_kind_head.weight) == 0)
    assert np.all(np.asarray(migrated_adam.nu.build_kind_head.weight) == 0)


def _fake_refresh(count: int, repetitions: int, iteration: int):
    masks = np.ones((count, 3970), dtype=np.bool_)
    delta = np.resize(np.asarray([0.8, -0.8, 0.0], dtype=np.float32), count)
    successor_target = np.empty((count, repetitions, 2), dtype=np.float32)
    successor_target[:, :, 0] = -1.0
    successor_target[:, :, 1] = 1.0
    data = {
        "pre_observation": np.zeros((count, 2, 2, 2), dtype=np.float16),
        "pre_history": np.zeros((count, 2, 3), dtype=np.float16),
        "pre_mask": masks,
        "build_action": BUILD_START + np.arange(count, dtype=np.int32) % 441,
        "control_action": np.full(count, PASS_INDEX, dtype=np.int32),
        "delta_mean": delta,
        "delta_se": np.zeros(count, dtype=np.float32),
        "delta_shrunk": delta,
        "source_id": np.arange(count, dtype=np.uint64),
        "source_game_id": np.arange(count, dtype=np.uint64) // 2,
        "source_turn": np.arange(count, dtype=np.int32),
        "source_seat": np.arange(count, dtype=np.int8) % 2,
        "source_quota": np.arange(count, dtype=np.int8) % 3,
        "source_selection_probability": np.full(count, 0.5, dtype=np.float32),
        "proposal_method": np.arange(count, dtype=np.int8) % 4,
        "proposal_probability": np.full(count, 1.0, dtype=np.float32),
        "generator_iteration": np.full(count, iteration, dtype=np.int32),
        "successor_observation": np.zeros(
            (count, repetitions, 2, 2, 2, 2), dtype=np.float16
        ),
        "successor_history": np.zeros(
            (count, repetitions, 2, 2, 3), dtype=np.float16
        ),
        "successor_mask": np.ones(
            (count, repetitions, 2, 3970), dtype=np.bool_
        ),
        "successor_target": successor_target,
        "successor_finished": np.ones(
            (count, repetitions, 2), dtype=np.bool_
        ),
    }
    assert set(ACTOR_FIELDS) | set(SUCCESSOR_FIELDS) == set(data)
    return data


def test_buffer_batches_preserve_actor_quotas_and_successor_pairs(tmp_path):
    buffer = CastleCounterfactualBuffer(
        capacity=1000, max_age=100, repetitions=2, run_dir=tmp_path
    )
    buffer.add(_fake_refresh(96, 2, 3003), 3003)
    sampled = buffer.sample_epoch(
        current_iteration=3003,
        device_count=4,
        minibatches=3,
        seed=9,
        recent_fraction=0.75,
        actor_uniform_per_device=2,
        actor_positive_per_device=1,
        actor_negative_per_device=1,
        successor_per_device=4,
    )
    actor_delta = np.stack(
        [
            sampled["actor_cache"]["delta_shrunk"][device][
                sampled["actor_indices"][device]
            ]
            for device in range(4)
        ]
    )
    assert np.all((actor_delta > 0).sum(axis=2) >= 1)
    assert np.all((actor_delta < 0).sum(axis=2) >= 1)
    successor_target = np.stack(
        [
            sampled["successor_cache"]["successor_target"][device][
                sampled["successor_indices"][device]
            ]
            for device in range(4)
        ]
    )
    for step in range(3):
        global_targets = successor_target[:, step].reshape(-1)
        assert (global_targets == -1).sum() == 8
        assert (global_targets == 1).sum() == 8


def test_build_proposals_are_legal_distinct_and_stable():
    legal = np.zeros((2, 21, 21), dtype=np.bool_)
    legal[:, 0, :3] = True
    logits = np.zeros((2, 3970), dtype=np.float32)
    logits[:, BUILD_START : BUILD_START + 3] = [1.0, 3.0, 2.0]
    features = {
        "legal_builds": legal,
        "actor_logits": logits,
        "post_garrison": np.broadcast_to(
            np.arange(441).reshape(1, 21, 21), (2, 21, 21)
        ),
        "general_distance": np.broadcast_to(
            np.arange(441).reshape(1, 21, 21), (2, 21, 21)
        ),
    }
    first = propose_build_candidates(
        features, np.asarray([10, 11], dtype=np.uint64), 3003
    )
    second = propose_build_candidates(
        features, np.asarray([10, 11], dtype=np.uint64), 3003
    )
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    for source in range(2):
        actions = first["build_action"][first["source_row"] == source]
        assert len(np.unique(actions)) == len(actions)
        assert all(legal[source].reshape(-1)[action - BUILD_START] for action in actions)

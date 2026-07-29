"""Deterministic equivalence tests for the throughput refactor.

Each test pits the fused/compiled implementations in generals.training.train
against a reference re-implementation of the pre-refactor host-side pipeline,
on identical inputs. Runs on any device count (CPU CI has one device; the
cross-device pmean/psum math reduces to identity there).
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from generals.training.config import TrainingConfig
from generals.training.model import CompetitionTransformer
from generals.training.ppo import compute_gae, ppo_epoch
from generals.training.train import (
    _learning_rate,
    _learning_rate_float,
    _replicate_for_pmap,
    _save_checkpoint,
    make_ema_step,
    make_prepare_batch,
    make_update_shard,
)

DEVICES = jax.device_count()


def small_config(**overrides) -> TrainingConfig:
    defaults = dict(
        num_envs=4,
        num_steps=8,
        minibatch_size=4,
        advantage_top_fraction=0.25,
        ppo_epochs=1,
        use_bf16=False,
        value_bins=16,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def rollout_arrays(config: TrainingConfig, seed: int = 0):
    rng = np.random.default_rng(seed)
    seats = 2 * config.num_envs
    shape = (DEVICES, config.num_steps, seats)
    rewards = jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
    values = jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
    next_values = jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
    terminated = jnp.asarray(rng.random(shape) < 0.1)
    truncated = jnp.asarray(rng.random(shape) < 0.05) & ~terminated
    winners = jnp.asarray(
        rng.integers(-1, 2, size=shape), dtype=jnp.int32
    )
    return rewards, values, next_values, terminated, truncated, winners


def test_prepare_batch_matches_pre_refactor_pipeline():
    config = small_config()
    rewards, values, next_values, terminated, truncated, winners = rollout_arrays(
        config
    )
    kept_samples = int(
        2 * config.num_envs * config.num_steps * config.advantage_top_fraction
    )

    # --- reference: the pre-refactor pipeline, verbatim ---
    gae_pmapped = jax.pmap(
        lambda r, v, nv, t, tr: compute_gae(
            r, v, nv, t, tr, config.gamma, config.gae_lambda
        )
    )

    @partial(jax.pmap, axis_name="devices")
    def normalize_reference(advantages):
        mean = jax.lax.pmean(advantages.mean(), axis_name="devices")
        mean_square = jax.lax.pmean((advantages**2).mean(), axis_name="devices")
        standard_deviation = jnp.sqrt(jnp.maximum(mean_square - mean**2, 0.0))
        return (advantages - mean) / (standard_deviation + 1e-8)

    @jax.pmap
    def select_reference(advantages):
        _, indices = jax.lax.top_k(jnp.abs(advantages.reshape(-1)), kept_samples)
        return indices

    ref_advantages = gae_pmapped(rewards, values, next_values, terminated, truncated)
    ref_returns = ref_advantages + values
    ref_raw_std = float(ref_advantages.std())
    ref_normalized = normalize_reference(ref_advantages)
    ref_indices = select_reference(ref_normalized)

    done = terminated | truncated
    ref_episodes = int(done[:, :, : config.num_envs].sum())
    p0_terminated = terminated[:, :, : config.num_envs]
    p0_winners = winners[:, :, : config.num_envs]
    ref_wins = int(jnp.sum(p0_terminated & (p0_winners == 0)))
    ref_losses = int(jnp.sum(p0_terminated & (p0_winners == 1)))
    ref_return_variance = float(jnp.var(ref_returns))
    ref_ev = 1.0 - float(jnp.var(ref_returns - values)) / max(
        ref_return_variance, 1e-8
    )
    ref_mean_reward = float(rewards.mean())

    # --- fused implementation under test ---
    prepare_batch = make_prepare_batch(config)
    normalized, returns, indices, prep_metrics = prepare_batch(
        rewards, values, next_values, terminated, truncated, winners
    )
    prep_metrics = jax.device_get(prep_metrics)
    prep_metrics = {name: value[0] for name, value in prep_metrics.items()}

    np.testing.assert_allclose(normalized, ref_normalized, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(returns, ref_returns, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(indices, ref_indices)
    assert int(prep_metrics["episodes"]) == ref_episodes
    assert int(prep_metrics["wins"]) == ref_wins
    assert int(prep_metrics["losses"]) == ref_losses
    np.testing.assert_allclose(
        float(prep_metrics["raw_advantage_std"]), ref_raw_std, rtol=1e-5
    )
    np.testing.assert_allclose(
        float(prep_metrics["explained_variance"]), ref_ev, rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(
        float(prep_metrics["mean_reward"]), ref_mean_reward, rtol=1e-5, atol=1e-7
    )


def test_ema_step_matches_eager_reference():
    config = small_config()
    rng = np.random.default_rng(1)
    tree = {
        "weight": jnp.asarray(rng.normal(size=(5, 3)), dtype=jnp.float32),
        "bias": jnp.asarray(rng.normal(size=(3,)), dtype=jnp.float32),
    }
    current = {
        "weight": jnp.asarray(rng.normal(size=(5, 3)), dtype=jnp.float32),
        "bias": jnp.asarray(rng.normal(size=(3,)), dtype=jnp.float32),
    }

    ref = jax.tree.map(
        lambda ema, cur: config.ema_decay * ema + (1.0 - config.ema_decay) * cur,
        tree,
        current,
    )

    ema_step = make_ema_step(config)
    updated = ema_step(_replicate_for_pmap(tree), _replicate_for_pmap(current))
    for name in tree:
        # every replica holds the same values, equal to the eager result
        for device in range(DEVICES):
            np.testing.assert_allclose(
                updated[name][device], ref[name], rtol=1e-6, atol=1e-7
            )


def _tiny_network_and_batch(config: TrainingConfig, seed: int = 2):
    network = CompetitionTransformer(
        depth=1,
        model_dim=32,
        heads=4,
        ff_factor=2,
        value_bins=config.value_bins,
        use_bf16=False,
        key=jax.random.PRNGKey(seed),
    )
    rng = np.random.default_rng(seed)
    steps, envs = 2, 2
    total = steps * envs
    shape = (DEVICES, steps, envs)
    observations = jnp.asarray(
        rng.normal(size=(*shape, 38, 21, 21)), dtype=jnp.float32
    )
    histories = jnp.asarray(rng.normal(size=(*shape, 2, 512)), dtype=jnp.float32)
    masks = jnp.ones((*shape, 3970), dtype=jnp.bool_)
    actions = jnp.asarray(rng.integers(0, 3970, size=shape), dtype=jnp.int32)
    old_log_probs = jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
    advantages = jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
    returns = jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
    batch = (observations, histories, masks, actions, old_log_probs, advantages, returns)
    indices = jnp.broadcast_to(jnp.arange(total, dtype=jnp.int32), (DEVICES, total))
    return network, batch, indices


def test_update_shard_matches_pre_refactor_update():
    config = small_config(minibatch_size=2)
    network, batch, indices = _tiny_network_and_batch(config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(partial(_learning_rate, config)),
    )
    parameters, static = eqx.partition(network, eqx.is_inexact_array)
    optimizer_state = optimizer.init(parameters)
    parameters = _replicate_for_pmap(parameters)
    optimizer_state = _replicate_for_pmap(optimizer_state)
    device_keys = jax.random.split(jax.random.PRNGKey(7), DEVICES)
    entropy = np.full((DEVICES,), 0.01, dtype=np.float32)

    # --- reference: pre-refactor external key splitting + ppo_epoch pmap ---
    split_keys = jax.vmap(jax.random.split)(device_keys)
    ref_next_keys, update_keys = split_keys[:, 0], split_keys[:, 1]

    @partial(jax.pmap, axis_name="devices")
    def reference_update(params, opt_state, batch, idx, rng, entropy_coefficient):
        shard_network = eqx.combine(params, static)
        shard_network, opt_state, metrics = ppo_epoch(
            shard_network,
            opt_state,
            batch,
            idx,
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

    ref_params, _, ref_metrics = reference_update(
        parameters, optimizer_state, batch, indices, update_keys, entropy
    )

    # --- implementation under test: RNG split happens inside ---
    update_shard = make_update_shard(config, static, optimizer)
    new_params, _, next_keys, metrics = update_shard(
        parameters, optimizer_state, batch, indices, device_keys, entropy
    )

    np.testing.assert_array_equal(next_keys, ref_next_keys)
    for ref_leaf, new_leaf in zip(
        jax.tree.leaves(eqx.filter(ref_params, eqx.is_inexact_array)),
        jax.tree.leaves(eqx.filter(new_params, eqx.is_inexact_array)),
    ):
        np.testing.assert_array_equal(ref_leaf, new_leaf)
    for name in ref_metrics:
        np.testing.assert_allclose(
            metrics[name], ref_metrics[name], rtol=1e-6, atol=1e-7
        )


def test_checkpoint_roundtrip_with_replicated_ema(tmp_path):
    config = small_config()
    network = CompetitionTransformer(
        depth=1,
        model_dim=32,
        heads=4,
        ff_factor=2,
        value_bins=config.value_bins,
        use_bf16=False,
        key=jax.random.PRNGKey(3),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(partial(_learning_rate, config)),
    )
    optimizer_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    ema_parameters, static = eqx.partition(network, eqx.is_inexact_array)
    ema_replicated = _replicate_for_pmap(ema_parameters)

    # save the way the training loop now does: slice shard 0 back out
    ema_network = eqx.combine(
        jax.tree.map(lambda value: value[0], ema_replicated), static
    )
    key = jax.random.PRNGKey(4)
    path = tmp_path / "checkpoint.eqx"
    _save_checkpoint(path, network, optimizer_state, ema_network, 7, 1, key)

    skeleton = (network, optimizer_state, ema_network, jnp.int32(0), jnp.int32(0), key)
    loaded_network, _, loaded_ema, iteration, stage, _ = eqx.tree_deserialise_leaves(
        path, skeleton
    )
    assert int(iteration) == 7
    assert int(stage) == 1
    for original, loaded in zip(
        jax.tree.leaves(eqx.filter(network, eqx.is_inexact_array)),
        jax.tree.leaves(eqx.filter(loaded_network, eqx.is_inexact_array)),
    ):
        np.testing.assert_array_equal(original, loaded)
    loaded_ema_params, _ = eqx.partition(loaded_ema, eqx.is_inexact_array)
    for original, loaded in zip(
        jax.tree.leaves(ema_parameters), jax.tree.leaves(loaded_ema_params)
    ):
        np.testing.assert_array_equal(original, loaded)
    # and the load-side replication restores the on-device layout
    re_replicated = _replicate_for_pmap(loaded_ema_params)
    for leaf in jax.tree.leaves(re_replicated):
        assert leaf.shape[0] == DEVICES


@pytest.mark.parametrize("optimizer_step", [0, 128, 4096, 1_000_000])
def test_python_learning_rate_matches_jax_schedule(optimizer_step):
    config = small_config()
    np.testing.assert_allclose(
        _learning_rate_float(config, optimizer_step),
        float(_learning_rate(config, optimizer_step)),
        rtol=1e-6,
    )

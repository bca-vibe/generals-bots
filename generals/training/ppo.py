"""GAE, HL-Gauss value targets, and a data-parallel PPO epoch."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def compute_gae(
    rewards,
    values,
    next_values,
    terminated,
    truncated,
    gamma: float,
    gae_lambda: float,
):
    """Compute GAE, treating the competition's 1200-turn draw as terminal."""
    finished = terminated | truncated

    def step(carry, inputs):
        reward, value, next_value, done = inputs
        nonterminal = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * next_value * nonterminal - value
        advantage = delta + gamma * gae_lambda * nonterminal * carry
        return advantage, advantage

    _, reversed_advantages = jax.lax.scan(
        step,
        jnp.zeros_like(rewards[0]),
        (rewards[::-1], values[::-1], next_values[::-1], finished[::-1]),
    )
    return reversed_advantages[::-1]


def hl_gauss_cross_entropy(
    logits,
    target,
    *,
    num_bins: int,
    value_min: float,
    value_max: float,
    sigma: float,
):
    centers = jnp.linspace(value_min, value_max, num_bins)
    half_bin = (value_max - value_min) / (num_bins - 1) / 2.0
    upper = (centers + half_bin - target) / sigma
    lower = (centers - half_bin - target) / sigma
    probabilities = jax.scipy.special.ndtr(upper) - jax.scipy.special.ndtr(lower)
    probabilities = probabilities / jnp.maximum(probabilities.sum(), 1e-8)
    return -jnp.sum(probabilities * jax.nn.log_softmax(logits))


def ppo_epoch(
    network,
    optimizer_state,
    batch,
    sample_indices,
    optimizer,
    key,
    *,
    minibatch_size: int,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
    value_bins: int,
    value_min: float,
    value_max: float,
    hl_gauss_sigma: float,
    axis_name: str,
):
    """Run one shuffled PPO epoch on a single accelerator shard."""
    observations, histories, masks, actions, old_log_probs, advantages, returns = batch
    total = observations.shape[0] * observations.shape[1]
    observations = observations.reshape(total, *observations.shape[2:])
    histories = histories.reshape(total, *histories.shape[2:])
    masks = masks.reshape(total, *masks.shape[2:])
    actions = actions.reshape(total)
    old_log_probs = old_log_probs.reshape(total)
    advantages = advantages.reshape(total)
    returns = returns.reshape(total)

    permutation = jax.random.permutation(key, sample_indices.shape[0])
    minibatches = sample_indices[permutation].reshape(-1, minibatch_size)

    def minibatch_step(carry, indices):
        current_network, current_optimizer_state = carry
        minibatch = (
            observations[indices],
            histories[indices],
            masks[indices],
            actions[indices],
            old_log_probs[indices],
            advantages[indices],
            returns[indices],
        )

        def loss_fn(candidate):
            def sample_loss(obs, history, mask, action, old_log_prob, advantage, target):
                _, _, _, log_prob, entropy, value_logits = candidate(
                    obs, history, mask, None, action
                )
                log_ratio = log_prob - old_log_prob
                ratio = jnp.exp(log_ratio)
                unclipped = ratio * advantage
                clipped = jnp.clip(
                    ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
                ) * advantage
                policy_loss = -jnp.minimum(unclipped, clipped)
                value_loss = hl_gauss_cross_entropy(
                    value_logits,
                    target,
                    num_bins=value_bins,
                    value_min=value_min,
                    value_max=value_max,
                    sigma=hl_gauss_sigma,
                )
                total_loss = (
                    policy_loss
                    + value_coefficient * value_loss
                    - entropy_coefficient * entropy
                )
                approximate_kl = ratio - 1.0 - log_ratio
                clip_fraction = (jnp.abs(ratio - 1.0) > clip_epsilon).astype(jnp.float32)
                return total_loss, (
                    policy_loss,
                    value_loss,
                    entropy,
                    approximate_kl,
                    clip_fraction,
                )

            losses, auxiliary = jax.vmap(sample_loss)(*minibatch)
            policy_loss, value_loss, entropy, approximate_kl, clip_fraction = auxiliary
            metrics = {
                "loss": losses.mean(),
                "policy_loss": policy_loss.mean(),
                "value_loss": value_loss.mean(),
                "entropy": entropy.mean(),
                "approximate_kl": approximate_kl.mean(),
                "clip_fraction": clip_fraction.mean(),
            }
            return losses.mean(), metrics

        (_, metrics), gradients = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            current_network
        )
        gradients = jax.lax.pmean(gradients, axis_name=axis_name)
        gradient_leaves = jax.tree.leaves(eqx.filter(gradients, eqx.is_inexact_array))
        metrics["gradient_norm"] = jnp.sqrt(
            sum(jnp.sum(leaf**2) for leaf in gradient_leaves)
        )
        updates, current_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current_network
        )
        current_network = eqx.apply_updates(current_network, updates)
        return (current_network, current_optimizer_state), metrics

    (network, optimizer_state), metrics = jax.lax.scan(
        minibatch_step, (network, optimizer_state), minibatches
    )
    metrics = {name: values.mean() for name, values in metrics.items()}
    return network, optimizer_state, metrics

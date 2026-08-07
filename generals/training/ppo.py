"""GAE, HL-Gauss value targets, and a data-parallel PPO epoch."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .castle_exploration import apply_tactical_build_logit_boost


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
    tactical_build_logit_boost,
    axis_name: str,
):
    """Run one shuffled PPO epoch on a single accelerator shard."""
    (
        observations,
        histories,
        masks,
        actions,
        old_log_probs,
        advantages,
        returns,
        tactical_build_masks,
    ) = batch
    total = observations.shape[0] * observations.shape[1]
    observations = observations.reshape(total, *observations.shape[2:])
    histories = histories.reshape(total, *histories.shape[2:])
    masks = masks.reshape(total, *masks.shape[2:])
    actions = actions.reshape(total)
    old_log_probs = old_log_probs.reshape(total)
    advantages = advantages.reshape(total)
    returns = returns.reshape(total)
    tactical_build_masks = tactical_build_masks.reshape(total, *tactical_build_masks.shape[2:])

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
            tactical_build_masks[indices],
        )

        def loss_fn(candidate):
            def sample_loss(
                obs,
                history,
                mask,
                action,
                old_log_prob,
                advantage,
                target,
                tactical_build_mask,
            ):
                logits, _, value_logits = candidate.forward(obs, history, mask)
                behavior_logits = apply_tactical_build_logit_boost(
                    logits,
                    tactical_build_mask,
                    tactical_build_logit_boost,
                )
                log_prob = jax.nn.log_softmax(behavior_logits)[action]
                probabilities = jax.nn.softmax(logits)
                log_probabilities = jax.nn.log_softmax(logits)
                entropy = -jnp.sum(probabilities * log_probabilities)
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


def counterfactual_ppo_epoch(
    network,
    optimizer_state,
    batch,
    sample_indices,
    counterfactual,
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
    actor_coefficient: float,
    successor_coefficient: float,
    delta_coefficient: float,
    actor_temperature: float,
    actor_weight_scale: float,
    huber_delta: float,
    actor_coefficient_scale,
    critic_coefficient_scale,
    axis_name: str,
):
    """Run PPO with strictly separate supervised intervention minibatches."""
    (
        observations,
        histories,
        masks,
        actions,
        old_log_probs,
        advantages,
        returns,
        _tactical_build_masks,
    ) = batch
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
    actor_cache = counterfactual["actor_cache"]
    successor_cache = counterfactual["successor_cache"]
    actor_indices = counterfactual["actor_indices"]
    successor_indices = counterfactual["successor_indices"]
    inverse_probability = counterfactual["actor_inverse_probability"]
    gradient_diagnostics_enabled = counterfactual[
        "gradient_diagnostics_enabled"
    ]
    if actor_indices.shape[0] != minibatches.shape[0]:
        raise ValueError("Counterfactual actor batches do not match PPO minibatches")
    if successor_indices.shape[0] != minibatches.shape[0]:
        raise ValueError("Counterfactual successor batches do not match PPO minibatches")

    def minibatch_step(carry, inputs):
        current_network, current_optimizer_state = carry
        (
            ppo_indices,
            actor_index,
            actor_inverse,
            successor_index,
            step_index,
        ) = inputs
        ppo_minibatch = (
            observations[ppo_indices],
            histories[ppo_indices],
            masks[ppo_indices],
            actions[ppo_indices],
            old_log_probs[ppo_indices],
            advantages[ppo_indices],
            returns[ppo_indices],
        )
        actor = {name: value[actor_index] for name, value in actor_cache.items()}
        successor = {
            name: value[successor_index] for name, value in successor_cache.items()
        }

        def loss_fn(candidate):
            def ppo_sample_loss(obs, history, mask, action, old_log_prob, advantage, target):
                logits, _, value_logits = candidate.forward(obs, history, mask)
                log_prob = jax.nn.log_softmax(logits)[action]
                probabilities = jax.nn.softmax(logits)
                log_probabilities = jax.nn.log_softmax(logits)
                entropy = -jnp.sum(probabilities * log_probabilities)
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
                clip_fraction = (
                    jnp.abs(ratio - 1.0) > clip_epsilon
                ).astype(jnp.float32)
                return total_loss, (
                    policy_loss,
                    value_loss,
                    entropy,
                    approximate_kl,
                    clip_fraction,
                )

            ppo_losses, ppo_aux = jax.vmap(ppo_sample_loss)(*ppo_minibatch)
            policy_loss, value_loss, entropy, approximate_kl, clip_fraction = ppo_aux

            def actor_sample(
                obs,
                history,
                mask,
                build,
                control,
                delta_shrunk,
                delta_mean,
                correction,
            ):
                (
                    logits,
                    _,
                    _,
                    difference,
                    build_kind_residual,
                ) = candidate.forward_with_build_difference(obs, history, mask)
                margin = logits[build] - logits[control]
                probabilities = jax.nn.softmax(logits)
                build_logits = logits[8 * 21 * 21 : 9 * 21 * 21]
                build_probability = probabilities[8 * 21 * 21 : 9 * 21 * 21].sum()
                build_rank = 1 + jnp.sum(logits > logits[build])
                target = jax.nn.sigmoid(delta_shrunk / actor_temperature)
                preference_loss = -(
                    target * jax.nn.log_sigmoid(margin)
                    + (1.0 - target) * jax.nn.log_sigmoid(-margin)
                )
                magnitude = jnp.clip(
                    jnp.abs(delta_shrunk) / actor_weight_scale, 0.0, 1.0
                )
                weight = magnitude * correction
                cell = build - 8 * 21 * 21
                conditional_build_probability = jax.nn.softmax(build_logits)[cell]
                conditional_build_rank = 1 + jnp.sum(
                    build_logits > build_logits[cell]
                )
                predicted_delta = jnp.nan_to_num(difference).reshape(-1)[cell]
                residual = predicted_delta - delta_mean
                absolute = jnp.abs(residual)
                delta_loss = jnp.where(
                    absolute <= huber_delta,
                    0.5 * residual**2,
                    huber_delta * (absolute - 0.5 * huber_delta),
                )
                return weight * preference_loss, delta_loss, (
                    margin,
                    predicted_delta,
                    weight,
                    build_probability,
                    build_rank,
                    build_kind_residual,
                    conditional_build_probability,
                    conditional_build_rank,
                )

            actor_losses, delta_losses, actor_aux = jax.vmap(actor_sample)(
                actor["pre_observation"],
                actor["pre_history"],
                actor["pre_mask"],
                actor["build_action"],
                actor["control_action"],
                actor["delta_shrunk"],
                actor["delta_mean"],
                actor_inverse,
            )
            (
                actor_margin,
                predicted_delta,
                actor_weight,
                actor_build_probability,
                actor_build_rank,
                actor_build_kind_residual,
                actor_conditional_build_probability,
                actor_conditional_build_rank,
            ) = actor_aux

            def successor_sample(obs, history, mask, target):
                _, _, value_logits = candidate.forward(obs, history, mask)
                return hl_gauss_cross_entropy(
                    value_logits,
                    target,
                    num_bins=value_bins,
                    value_min=value_min,
                    value_max=value_max,
                    sigma=hl_gauss_sigma,
                )

            successor_losses = jax.vmap(successor_sample)(
                successor["successor_observation"],
                successor["successor_history"],
                successor["successor_mask"],
                successor["successor_target"],
            )
            ppo_loss = ppo_losses.mean()
            actor_loss = actor_losses.mean()
            successor_loss = successor_losses.mean()
            delta_loss = delta_losses.mean()
            effective_actor = actor_coefficient_scale * actor_coefficient
            effective_successor = critic_coefficient_scale * successor_coefficient
            effective_delta = critic_coefficient_scale * delta_coefficient
            total_loss = (
                ppo_loss
                + effective_actor * actor_loss
                + effective_successor * successor_loss
                + effective_delta * delta_loss
            )
            positive = actor["delta_shrunk"] > 0
            negative = actor["delta_shrunk"] < 0

            def masked_mean(value, condition):
                return jnp.where(condition, value, 0.0).sum() / jnp.maximum(
                    condition.sum(), 1
                )

            metrics = {
                "loss": total_loss,
                "ppo_total_loss": ppo_loss,
                "policy_loss": policy_loss.mean(),
                "value_loss": value_loss.mean(),
                "entropy": entropy.mean(),
                "approximate_kl": approximate_kl.mean(),
                "clip_fraction": clip_fraction.mean(),
                "counterfactual_actor_loss": actor_loss,
                "counterfactual_successor_value_loss": successor_loss,
                "counterfactual_delta_loss": delta_loss,
                "counterfactual_actor_weight_mean": actor_weight.mean(),
                "counterfactual_actor_margin_positive": masked_mean(
                    actor_margin, positive
                ),
                "counterfactual_actor_margin_negative": masked_mean(
                    actor_margin, negative
                ),
                "counterfactual_delta_prediction_positive": masked_mean(
                    predicted_delta, positive
                ),
                "counterfactual_delta_prediction_negative": masked_mean(
                    predicted_delta, negative
                ),
                "counterfactual_build_probability_positive": masked_mean(
                    actor_build_probability, positive
                ),
                "counterfactual_build_probability_negative": masked_mean(
                    actor_build_probability, negative
                ),
                "counterfactual_build_rank_positive": masked_mean(
                    actor_build_rank, positive
                ),
                "counterfactual_build_rank_negative": masked_mean(
                    actor_build_rank, negative
                ),
                "counterfactual_build_kind_residual_positive": masked_mean(
                    actor_build_kind_residual, positive
                ),
                "counterfactual_build_kind_residual_negative": masked_mean(
                    actor_build_kind_residual, negative
                ),
                "counterfactual_conditional_build_probability_positive": masked_mean(
                    actor_conditional_build_probability, positive
                ),
                "counterfactual_conditional_build_probability_negative": masked_mean(
                    actor_conditional_build_probability, negative
                ),
                "counterfactual_conditional_build_rank_positive": masked_mean(
                    actor_conditional_build_rank, positive
                ),
                "counterfactual_conditional_build_rank_negative": masked_mean(
                    actor_conditional_build_rank, negative
                ),
                "counterfactual_actor_coefficient_scale": actor_coefficient_scale,
                "counterfactual_critic_coefficient_scale": critic_coefficient_scale,
                "counterfactual_actor_coefficient_effective": effective_actor,
                "counterfactual_value_coefficient_effective": effective_successor,
                "counterfactual_delta_coefficient_effective": effective_delta,
            }
            return total_loss, metrics

        (_, metrics), gradients = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(current_network)
        gradients = jax.lax.pmean(gradients, axis_name=axis_name)
        gradient_leaves = jax.tree.leaves(
            eqx.filter(gradients, eqx.is_inexact_array)
        )
        metrics["gradient_norm"] = jnp.sqrt(
            sum(jnp.sum(leaf**2) for leaf in gradient_leaves)
        )

        def component_gradient_diagnostics(_):
            def component(name, coefficient=1.0):
                gradient = eqx.filter_grad(
                    lambda candidate: coefficient * loss_fn(candidate)[1][name]
                )(current_network)
                return jax.lax.pmean(gradient, axis_name=axis_name)

            ppo_gradient = component("ppo_total_loss")
            actor_gradient = component(
                "counterfactual_actor_loss",
                actor_coefficient_scale * actor_coefficient,
            )
            successor_gradient = component(
                "counterfactual_successor_value_loss",
                critic_coefficient_scale * successor_coefficient,
            )
            delta_gradient = component(
                "counterfactual_delta_loss",
                critic_coefficient_scale * delta_coefficient,
            )
            cf_gradient = jax.tree.map(
                lambda actor, successor, delta: actor + successor + delta,
                actor_gradient,
                successor_gradient,
                delta_gradient,
            )

            def norm(tree):
                leaves = jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))
                return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))

            def dot(left, right):
                left_leaves = jax.tree.leaves(
                    eqx.filter(left, eqx.is_inexact_array)
                )
                right_leaves = jax.tree.leaves(
                    eqx.filter(right, eqx.is_inexact_array)
                )
                return sum(
                    jnp.sum(a * b)
                    for a, b in zip(left_leaves, right_leaves, strict=True)
                )

            ppo_norm = norm(ppo_gradient)
            cf_norm = norm(cf_gradient)
            return {
                "diagnostic_ppo_gradient_norm": ppo_norm,
                "diagnostic_actor_cf_gradient_norm": norm(actor_gradient),
                "diagnostic_build_kind_actor_gradient_norm": norm(
                    actor_gradient.build_kind_head
                ),
                "diagnostic_successor_cf_gradient_norm": norm(successor_gradient),
                "diagnostic_delta_cf_gradient_norm": norm(delta_gradient),
                "diagnostic_total_cf_gradient_norm": cf_norm,
                "diagnostic_cf_to_ppo_gradient_ratio": cf_norm
                / jnp.maximum(ppo_norm, 1e-12),
                "diagnostic_ppo_cf_gradient_cosine": dot(
                    ppo_gradient, cf_gradient
                )
                / jnp.maximum(ppo_norm * cf_norm, 1e-12),
            }

        diagnostic_metrics = jax.lax.cond(
            gradient_diagnostics_enabled & (step_index == 0),
            component_gradient_diagnostics,
            lambda _: {
                "diagnostic_ppo_gradient_norm": jnp.asarray(0.0),
                "diagnostic_actor_cf_gradient_norm": jnp.asarray(0.0),
                "diagnostic_build_kind_actor_gradient_norm": jnp.asarray(0.0),
                "diagnostic_successor_cf_gradient_norm": jnp.asarray(0.0),
                "diagnostic_delta_cf_gradient_norm": jnp.asarray(0.0),
                "diagnostic_total_cf_gradient_norm": jnp.asarray(0.0),
                "diagnostic_cf_to_ppo_gradient_ratio": jnp.asarray(0.0),
                "diagnostic_ppo_cf_gradient_cosine": jnp.asarray(0.0),
            },
            operand=None,
        )
        metrics.update(diagnostic_metrics)
        updates, current_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current_network
        )
        current_network = eqx.apply_updates(current_network, updates)
        return (current_network, current_optimizer_state), metrics

    (network, optimizer_state), metrics = jax.lax.scan(
        minibatch_step,
        (network, optimizer_state),
        (
            minibatches,
            actor_indices,
            inverse_probability,
            successor_indices,
            jnp.arange(minibatches.shape[0]),
        ),
    )
    metrics = {
        name: values.max() if name.startswith("diagnostic_") else values.mean()
        for name, values in metrics.items()
    }
    return network, optimizer_state, metrics

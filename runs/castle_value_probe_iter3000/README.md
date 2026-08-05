# Castle successor-value probe — checkpoint 3000

## Conclusion

The value function does **not** believe castle construction is value-increasing,
including for castle builds that are beneficial under paired stochastic
continuations. It has learned a strong immediate debit for the armies spent on
construction, and only a weak ordinal signal about which builds are less bad.
It has not learned to credit the castle's future production enough to reverse
the sign in genuinely favorable states.

This is a critic/credit-assignment failure in addition to the previously
observed policy-logit failure. Because training uses only terminal `+1/-1`
rewards and `gamma = 1`, a correct successor value should include the castle's
full undiscounted strategic benefit.

## Experiment

- Checkpoint: iteration 3000 EMA, SHA-256
  `bac6cf4029eac926673109d3a282887007f47eeecb575318e1b236780cad6256`.
- Source: stochastic self-play on the final-stage competition map generator.
- Opportunities: 2,016 legal-build states, stratified by 200-turn bins.
- At each state: force either the highest-logit legal build site or the
  highest-logit legal non-build action.
- Pairing: 16 repetitions with identical opponent action and common future
  categorical random numbers across branches.
- Probe: immediately after the forced action, evaluate the actor's recurrent
  value head on both successor states and compute
  `V(successor_build) - V(successor_control)`.
- Scale: the network value is an expectation on `[-1, 1]`. Outcome effects
  below use game score on `[0, 1]`.
- Exclusion: 291 intervention pairs terminated immediately; 17 states with no
  nonterminal successor pair were excluded from value comparisons.
- Uncertainty: confidence intervals cluster-bootstrap the source self-play
  game, not individual opportunity states.

## Results

Across 1,999 analyzable states, the mean successor value difference was
**-0.4724** (source-game-clustered 95% CI **[-0.4918, -0.4540]**). Only
**0.60%** of states had a positive mean build-versus-control value difference.
The build successor exceeded the state's pre-action value in only **0.50%** of
states. By comparison, the non-build successor was essentially unchanged from
the pre-action estimate on average (`+0.00075`).

The result is not explained by averaging in bad builds:

| Causal group | States | Paired score effect | Predicted successor value effect | Value effect > 0 |
|---|---:|---:|---:|---:|
| Full-sample positive | 624 | **+0.2494** | **-0.4583** | 0.32% |
| Positive in both 8-rollout halves | 373 | **+0.3585** | **-0.4788** | **0.00%** |
| Full-sample negative | 899 | -0.3010 | -0.6293 | 0.11% |
| Negative in both halves | 577 | -0.4222 | -0.7289 | 0.00% |

For the most stringent descriptive group—positive in both halves—the critic
lowered value on every single state, even though forcing the castle improved
game score by 35.85 percentage points on average.

A held-out test reaches the same conclusion. Classifying a state as good using
one 8-rollout half and measuring both causal effect and successor value on the
other half (then swapping halves) produced 1,124 state-half selections. Their
held-out causal score effect was **+0.2040**, while their predicted successor
value effect was **-0.4778** (clustered 95% CI **[-0.5004, -0.4560]**); only
**0.44%** were predicted positive.

## What the critic has learned

The value difference has a real but incomplete ranking signal. Its Spearman
correlation with causal effect is **+0.305**; cross-half correlations are
**+0.301** and **+0.273**. Favorable builds are therefore penalized somewhat
less than harmful builds: the mean prediction is -0.458 for positive states
versus -0.629 for negative states, a difference of +0.171 (clustered 95% CI
**[+0.136, +0.207]**).

That signal only identifies catastrophically bad builds, not profitable ones.
The critic's least-negative decile had predicted value effect -0.0199 but an
actual causal score effect of approximately zero (-0.0019). The worst decile
had predicted value effect -1.174 and actual score effect -0.443. Thus the
critic can reject the worst construction choices, but it does not supply a
positive advantage that PPO could use to discover good construction choices.

## Self-play game length

A separate sample of 4,096 categorical EMA self-play games on the same final
stage produced:

- Mean: **604.6 turns**
- Median: **595 turns**
- 10th / 25th / 75th / 90th percentiles: **240 / 384 / 817 / 985**
- Reached the 1,200-turn cap: **57 games (1.39%)**
- Decisive games only: mean **596.2**, median **591**

The same sample built a castle in only 6/4,096 games (0.146%), despite at least
one legal opportunity in 4,012 games.

## Implication

Continuing the current recipe asks PPO to overcome both an action probability
near zero and a critic that assigns a large negative one-step advantage to the
action even in causally good states. More iterations may improve the weak
ranking signal, but the present learning dynamics provide little mechanism for
crossing the sign barrier. This strengthens the case for explicit build-kind
exploration plus counterfactual value/advantage supervision (and longer credit
assignment), rather than a flat castle reward.

The same seed did not reproduce the original atlas's exact sampled trajectory
set—the stochastic self-play paths diverged slightly—so this is treated as a
fresh same-checkpoint replication. Its aggregate causal result was consistent
with the first atlas: forcing builds everywhere changed score by -5.80 points
(first atlas: -6.18 points).

## Artifacts and compute

- `value_analysis.json`: complete derived statistics.
- `selfplay_length_4096.json`: full self-play aggregate report.
- `atlas_with_values/atlas.json`: counterfactual aggregate report.
- `atlas_with_values/paired_rollouts.npz`: raw outcomes and successor values.
- `results.tar`: byte-identical H100 export, SHA-256
  `2972c7f72e8dee6b424600995d126781b44ee452cb1e16fefb58c4f245b41ffe`.
- `complete_results.tar`: local archive including this report and derived
  analysis.

The single-H100 run used 33.08 billed minutes and cost **$2.20335**. Node
`generals-castle-value` was stopped and verified `stopped (disk intact)` after
export.

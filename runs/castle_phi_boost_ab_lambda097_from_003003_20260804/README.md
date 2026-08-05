# Castle potential-shaping A/B from checkpoint 3,003

## Conclusion

The treatment did **not** teach a persistent castle-building policy and did not
outperform the control. Both branches became substantially stronger than the
iteration-3,003 parent, but their terminal playing strength was effectively a
tie. The treatment selected zero castles in the complete raw and EMA
checkpoint round robins, while the control produced a few rare builds.

The experiment therefore rejects this particular intervention: a small,
actor-only potential combined with a fixed tactical `+10` build-logit boost.
It does **not** establish that potential-based shaping or targeted exploration
cannot work. The follow-up analysis indicates that this version failed to give
the raw critic a castle-aware target and ultimately drove the underlying build
probability lower than the control.

## Question and design

The preceding audits found two coupled failures at checkpoint 3,000:

- legal builds existed in nearly every game, but build probability was near
  zero; and
- the critic predicted a large value loss after building, including in states
  where paired continuations showed that building increased win probability.

We tested whether longer credit assignment plus a temporary economic potential
and tactical action boost could escape that basin.

Both arms resumed the exact terminal checkpoint at global iteration 3,003:

- Source SHA-256:
  `d669c7fb28d530c5ba12e460c4e2e00b5cc5900fbdebf1da402b47e9745e8c72`
- Architecture: convolutional transformer, depth 7, width 448
- Curriculum: final competition stage
- Preserved state: raw weights, Adam moments/count, EMA weights, RNG, and
  `ema_decay=0.999`
- Four H100s per arm, 512 environments/GPU, 1,024 minibatch/GPU
- 2,097,152 player-steps per arm per iteration
- 1,000 continuation iterations, ending at global iteration 4,003
- 2,097,152,000 new player-steps per arm; 8,394,899,456 cumulative samples at
  the terminal checkpoints

The split restored the same global environment and minibatch sizes used on the
parent 8-GPU run. Both branches used `gamma=1.0`, `gae_lambda=0.97`, and
`advantage_top_fraction=0.25`. Raising lambda in **both** arms made the control
a new-λ control rather than an unchanged continuation from 3,003.

### Control

- Original terminal win/loss/draw reward only
- Original flat 3,970-action policy
- No potential shaping or build-action intervention

Exact recipe:
[`castle_ab_lambda097_control_from_3003.toml`](../../generals/training/configs/castle_ab_lambda097_control_from_3003.toml)

### Treatment

The treatment added two mechanisms on top of the same λ=0.97 continuation.

1. An actor-only, zero-sum potential bounded to `[-0.10, 0.10]`, composed of:
   - `0.05 * remaining-land-horizon * tanh(land_margin / 20)`; and
   - a `0.05` risk-adjusted castle-asset term using remaining production,
     disadvantage, garrison, and distance from enemy land.
2. A `+10` additive behavior-logit boost at legal build sites satisfying the
   tactical gate: behind in army or land, post-build reserve at least 10, no
   visible/recent enemy within three steps, and sufficient production time to
   repay cost plus 25.

The intervention was full through global iteration 3,600, annealed to zero by
3,800, and remained zero for the final 203 iterations. Evaluation and
competition inference always used the underlying policy without the boost.
Stored and recomputed PPO log probabilities both used the exact boosted
behavior distribution.

The critic continued to target unshaped terminal returns. Actor GAE used the
potential-shaped reward, but the value head itself was deliberately left with
its original win-probability meaning.

Exact recipe:
[`castle_ab_lambda097_phi_boost_from_3003.toml`](../../generals/training/configs/castle_ab_lambda097_phi_boost_from_3003.toml)

Implementation:
[`potential.py`](../../generals/training/potential.py) and
[`castle_exploration.py`](../../generals/training/castle_exploration.py)

## Tracking and publication

- W&B group: `castle_phi_boost_ab_lambda097_from_003003_20260804`
- [Control W&B run](https://wandb.ai/bcarnold-independent/generals-bots/runs/castle-ab-lambda097-control-from-003003-20260804)
- [Treatment W&B run](https://wandb.ai/bcarnold-independent/generals-bots/runs/castle-ab-lambda097-phi-boost-from-003003-20260804)
- Hugging Face control root:
  `runs/castle_phi_boost_ab_lambda097_from_003003_20260804/control_lambda097`
- Hugging Face treatment root:
  `runs/castle_phi_boost_ab_lambda097_from_003003_20260804/treatment_phi_boost`

Archival checkpoints were produced at 3,200, 3,400, 3,600, 3,800, and 4,000,
plus terminal 4,003. Publications contain the full raw/Adam/EMA checkpoint and
both raw and EMA competition bundles. The terminal full-checkpoint hashes are:

- Control 4,003:
  `1f57165e7dd291e2776603457022751297ef9e95f197a8f08733dce38ef0e998`
- Treatment 4,003:
  `b722526d792f494dd45fc31eb405cb0b7e30afbda4336bedcd611742ca06a97d`

## Playing-strength results

The terminal evaluation played every archived checkpoint, the parent, and
both policies in a 13-participant round robin. Each participant played 3,072
games per policy type. Scores include draws as one half.

| Iteration | Control raw | Treatment raw | Control EMA | Treatment EMA |
|---:|---:|---:|---:|---:|
| 3,200 | 39.96% | 42.06% | 32.23% | 33.15% |
| 3,400 | 38.00% | 38.26% | 42.01% | 41.18% |
| 3,600 | 50.36% | 51.51% | 50.16% | 50.65% |
| 3,800 | 53.40% | 54.04% | 57.00% | 58.97% |
| 4,000 | 59.33% | 58.69% | 65.92% | 65.87% |
| 4,003 | **57.81%** | **56.64%** | **64.89%** | **66.18%** |

Same-iteration head-to-head scores below are from the control's perspective:

| Iteration | Raw control score | EMA control score |
|---:|---:|---:|
| 3,200 | 47.46% | 49.80% |
| 3,400 | 48.83% | 48.63% |
| 3,600 | 49.22% | 44.53% |
| 3,800 | 54.69% | 44.14% |
| 4,000 | 48.05% | 50.59% |
| 4,003 | **48.44%** | **52.54%** |

At the terminal checkpoint, treatment raw narrowly beat control raw, while
control EMA narrowly beat treatment EMA. Neither difference supports a robust
treatment advantage. Both branches clearly improved over parent EMA-3,003;
the terminal EMA scores against the parent were 88.48% for control and 85.74%
for treatment in the round robin.

The heuristic league was almost saturated by both arms (roughly 99% macro at
the end), making branch-versus-parent and checkpoint round-robin results more
discriminating than the heuristic aggregate.

Full result:
[`round_robin.json`](round_robin.json)

## Castle behavior

The decisive behavioral result is that the treatment never retained building
after the intervention was removed.

Across each checkpoint's 3,072 round-robin player-games:

| Policy | Control castle builds | Treatment castle builds |
|---|---:|---:|
| Raw 3,200 / 3,400 / 3,600 / 3,800 / 4,000 / 4,003 | 0 / 1 / 0 / 0 / 0 / 2 | **0 / 0 / 0 / 0 / 0 / 0** |
| EMA 3,200 / 3,400 / 3,600 / 3,800 / 4,000 / 4,003 | 0 / 0 / 2 / 0 / 0 / 2 | **0 / 0 / 0 / 0 / 0 / 0** |

These are successful castles constructed by the named neural policy, not
opponent actions or merely selected/invalid build actions. Legal opportunities
numbered in the hundreds of thousands for every checkpoint.

The terminal treatment also built zero castles in 4,096 exploration-free raw
self-play games and zero in 4,096 EMA games, despite legal opportunities in
about 98% of games.

Control's rare builds are compatible with longer training and/or λ=0.97
occasionally reopening castle behavior, but the event count is too small to
claim a learned strategic rule.

## Post hoc critic diagnosis

The common-state follow-up compared raw checkpoints 3,000, control 4,000, and
treatment 4,000 on identical build/control successors and causal labels.

| Raw critic | Mean `V(build) - V(control)` | Stable-good builds | Fraction positive |
|---|---:|---:|---:|
| Parent 3,000 | -0.4434 | -0.4230 | 2.11% |
| Control 4,000 | **-0.3907** | **-0.3661** | **6.53%** |
| Treatment 4,000 | -0.4068 | -0.3973 | 3.32% |

Control improved the critic's response, although it remained badly negative.
Treatment improved less. On identical states, treatment minus control was
`-0.0161` overall (95% CI `[-0.0203, -0.0118]`) and `-0.0313` on stable-good
builds (95% CI `[-0.0399, -0.0224]`).

The treatment's median aggregate build probability at 4,000 was `2.53e-11`,
versus `3.49e-9` for control: about 138 times lower. Its median build margin
was also worse (`-24.29` versus `-19.25`). The treatment therefore learned to
suppress the underlying build policy strongly enough to counteract the
temporary behavior boost.

Detailed protocol, tables, confidence intervals, and plots:
[`../castle_raw_checkpoint_followup_20260804/README.md`](../castle_raw_checkpoint_followup_20260804/README.md)

## What we learned

1. **The higher lambda was safe and useful for continued learning.** Both arms
   improved markedly over parent 3,003; the control also showed modest critic
   movement toward valuing build successors less negatively.
2. **This potential was too small and indirect to solve the critic failure.**
   It was bounded to ±0.10, acted only in actor GAE, and never trained the raw
   value head on a shaped target.
3. **A large fixed logit boost guarantees behavior samples, not retention.**
   Once the boost disappeared, the underlying treatment policy was even less
   likely to build than control.
4. **More castles is not itself the objective.** The earlier counterfactual
   atlas showed that forcing builds everywhere costs about 6.18 score points,
   while a selective observable subset is beneficial.
5. **The negative result narrows the next intervention.** Any follow-up should
   separate raw and shaped critic semantics, train a dedicated shaped critic,
   use an exactly accounted exploration mechanism that preserves gradients on
   underlying build probability, and reserve an intervention-free retention
   window.

## Completeness and compute

The intended 1,000 training iterations, periodic leagues, terminal 4,003
checkpoints, raw and EMA bundles, and full checkpoint round robin completed.
The optional large terminal 2,016×16 counterfactual rerun was intentionally
stopped because it repeated an already established diagnosis; the smaller raw
checkpoint follow-up supplied the common-state critic comparison used above.

The 8×H100 node ran for 266.48 billed minutes and cost **$141.98232**. It was
stopped with disk retained after artifacts were secured.

# Castle counterfactual atlas — checkpoint 3,000

## Conclusion

Checkpoint 3,000's EMA policy can exploit castles in a substantial, learnable
subset of ordinary self-play states, but its policy logits do not identify
those states. Continuing the current self-play recipe unchanged is therefore
unlikely to recover the behavior efficiently: the useful signal exists, but
the action is assigned essentially zero probability and the probability has no
measurable relationship to the action's causal value.

This is not evidence for building castles indiscriminately. Forcing a castle at
every legal opportunity reduced score by **6.18 percentage points** (state-
clustered 95% CI **[-7.65, -4.73]**). The important result is heterogeneity:
building is strongly useful in some contexts and strongly harmful in others.

## Experiment

- Full checkpoint iteration: 3,000, stage 4
- Checkpoint SHA-256:
  `bac6cf4029eac926673109d3a282887007f47eeecb575318e1b236780cad6256`
- Policy: EMA
- Maps and rules: final curriculum stage, exact competition generator
- Source distribution: checkpoint categorical self-play
- Source games: 1,024
- Legal opportunity player-steps observed: 411,550
- Filled per-game/seat/time-bin reservoirs: 6,549
- Selected opportunity states: 2,016
- Repetitions per state: 16
- Paired continuations: 32,256
- Total branch rollouts: 64,512
- All 64,512 branch rollouts finished

Each selected state was cloned. The control branch was forced to take the
model's highest-logit legal non-build action; the treatment branch was forced
to build at the model's highest-logit legal build site. Both branches then used
the EMA policy's categorical sampler on both seats. Corresponding branches used
identical categorical random keys, including for the opponent's intervention-
turn action.

Opportunity states were reservoir-sampled within each game, seat, and 200-turn
window. The final time distribution was 376 / 379 / 372 / 359 / 346 / 184
states from turns 0–199 through 1,000–1,199. The last window is smaller because
few games survive that long.

## Main causal results

| Intervention | Score | Wins | Draws | Losses |
|---|---:|---:|---:|---:|
| Best legal non-build | 50.58% | 15,350 | 1,932 | 14,974 |
| Best legal build | 44.40% | 13,638 | 1,370 | 17,248 |

The paired score difference is **-6.18 points**. It is negative in every broad
turn window, ranging from -8.91 points at turns 200–399 to -2.85 at 800–999.

The state-level effects are nevertheless stable:

- 648 states had a positive 16-rollout estimate, 913 negative, and 455 tied.
- The first- versus second-half state effects had Spearman correlation **0.712**.
- Their positive/non-positive classifications agreed **82.2%** of the time.
- Selecting states using one rollout half and scoring the choice on the other
  produced **+6.50 points overall** while selecting builds in about 29% of
  states. This is a per-state oracle diagnostic, not a deployable selector.

## Where builds help

The clearest pattern is that castles are a comeback/economic action, not a
win-more action under this continuation policy.

| State segment | States | Paired build effect |
|---|---:|---:|
| Army margin -49 to -1 | 624 | **+6.94 points** |
| Army margin <= -50 | 366 | **+2.72 points** |
| Army margin 0 to +49 | 623 | **-23.74 points** |
| Army margin >= +50 | 403 | **-7.43 points** |
| Land margin <= -10 | 597 | **+7.61 points** |
| Land margin 0 to +9 | 452 | **-19.44 points** |
| Land margin >= +10 | 466 | **-13.37 points** |
| True enemy-land distance 0–1 | 1,058 | **-12.02 points** |
| True enemy-land distance 4–6 | 256 | **+5.96 points** |
| True enemy-land distance >= 7 | 56 | **+12.67 points** |

Safety and liquidity matter. A zero-army post-build garrison cost 14.14 points;
the loss shrank to 3.32 points with a garrison of at least 25. Sites costing
41–46 (usually nearer an existing structure) were approximately neutral as a
group, while base-cost 35 sites lost 8.43 points, likely because base-cost sites
are often exposed frontier stacks.

Univariate rank correlations with causal build effect were -0.281 for land
margin, -0.264 for army margin, +0.189 for true enemy-land distance, and +0.093
for post-build garrison. The EMA value estimate correlated -0.293: states the
model considers worse are precisely the states where a castle is more likely to
help.

## Castle survival and payback

- Uninterrupted builder ownership: 66.3% at 25 turns, 58.9% at 50, 50.7% at
  100, and 42.7% at 200.
- Gross production paid back the build cost in 52.3% of rollouts; median
  mechanical payback was 70 turns.
- The builder owned the site at game end in 45.8% of forced-build rollouts.

"Gross payback" counts structure production while the builder owns the castle;
it does not assign tactical value to the castle or debit armies later moved
through the site.

## The logits are the failure

Across selected legal opportunities:

- Median aggregate build probability: **5.03e-8**
- Geometric mean aggregate build probability: **3.88e-8**
- Median best-build rank: **368**
- Median best-build logit margin versus best non-build: **-16.56**
- Spearman(log build probability, causal build effect): **0.017**
- Spearman(build logit margin, causal build effect): **0.024**

Mean build probability was slightly *lower* in positive-estimate states than in
negative-estimate states. The policy is not merely underconfident; its build
scores contain almost no useful ordering signal.

## Can observable features learn the useful subset?

A follow-up used only this atlas. Repetitions were split 8/8, and models were
5-fold cross-fitted by source game: train causal deltas came from one rollout
half, evaluation deltas from the other half on unseen source games, then the
halves were swapped. No true enemy geometry was included in the primary
feature set.

| Cross-fitted selector | Build rate | Conditional build effect | Overall uplift | Group-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Ridge | 34.6% | +8.72 points | **+3.02 points** | [+2.35, +3.72] |
| Histogram gradient boosting | 38.3% | +10.39 points | **+3.98 points** | [+3.19, +4.76] |
| Random forest | 35.3% | +11.49 points | **+4.06 points** | [+3.26, +4.86] |

Adding true enemy-land distance raised the best result only modestly, to about
+4.42 points. Most of the classification signal is therefore present in the
current policy observation or its own outputs: army/land totals, build cost,
garrison, structure distance, turn, value estimate, and logits.

These models are diagnostic, not deployable policies. Hyperparameters were
fixed and state/noise leakage was controlled, but the result still uses one
checkpoint, one atlas seed, and the same environment distribution. It should
be replicated on fresh maps and rollouts before using the magnitude as an
expected training gain.

## Training implication

The result rules out the pessimistic hypothesis that checkpoint 3,000 simply
cannot use a castle after one is built. It can: the current continuation policy
obtains large, repeatable gains in a learnable subset of states. The bottleneck
is discovering and assigning policy probability to the build action.

Unchanged training is unlikely to solve that bottleneck. At probabilities near
1e-8 to 1e-6, helpful build actions are almost never sampled, and the current
logit ordering provides no preference for the helpful subset. Terminal-only,
long-horizon credit then has little chance to move the head in the right
direction.

The most direct next experiment is therefore targeted but correctly accounted
build exploration (for example, a conditional action-kind floor) combined with
counterfactual build-value supervision from paired states like these. Train the
selector/value signal on observable features, preserve correct behavior-policy
log probabilities for PPO, and validate on fresh seeds. Do not add a flat
per-castle reward: the atlas shows that indiscriminate building is substantially
harmful.

## Limitations

- The treatment uses the model's favorite legal build site, not an exhaustive
  search over sites.
- Both branches use the checkpoint's own continuation policy. An improved
  post-build controller could make additional states positive.
- The control action is the model's best legal non-build action; this measures
  a concrete action-value comparison rather than build versus an average
  stochastic action.
- Segment rules are descriptive and post hoc. The cross-fitted selector is the
  stronger learnability check.

## Files

- `atlas.json`: aggregate report, segments, and extreme examples
- `paired_rollouts.npz`: per-state features and all paired outcomes
- `learnability.json`: grouped cross-fit and rule diagnostics
- `results.tar`: byte-identical export from the H100 before the local
  learnability follow-up

H100 node `generals-castle-atlas` was stopped and verified as
`stopped (disk intact)` after the artifacts were downloaded.

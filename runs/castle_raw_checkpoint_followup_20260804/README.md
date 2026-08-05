# Raw-checkpoint stochastic league and castle-critic follow-up

## Executive conclusion

Current self-play is not producing enough castle exploration to bootstrap a
reliable build policy. Across 7,680 temperature-1 stochastic games, only 13
castles were constructed, in 13 distinct games. Raw continuation 3,000 built
none in any matchup; treatment 4,000 built one across 3,072 policy-player
games and none in self-play. This is not caused by a lack of legal actions:
94.1%--96.1% of each policy's games contained at least one legal-build
opportunity.

The no-build policy is locally coherent but not globally complete. On 2,016
legal-build states, raw 3,000's highest-logit legal build reduced score by
0.0483 on average (source-game-clustered 95% CI -0.0623 to -0.0343). However,
413 states were causally positive in both independent 8-rollout halves, with
mean paired score gain +0.3411. The symmetric held-out 8/8 selector produced
+0.2071 score uplift. Thus useful castle states exist and are reproducible.

The critics do not recognize those states. On identical raw-3,000 build and
control successors, every checkpoint assigned a strongly negative value
difference even to held-out good builds. Control 4,000 moved in the right
direction relative to raw 3,000, but remained negative. The phi-boost
treatment improved less than control and was significantly more negative than
control on identical successors. Its policy also reduced median castle
probability by about 89x relative to raw 3,000.

The practical conclusion is that simply continuing the current self-play
recipe is unlikely to discover a robust castle policy on a useful time scale.
The data support targeted, correctly accounted action-kind exploration plus a
counterfactual/auxiliary critic target, not an unconditional castle reward.

## Checkpoints and protocol

All policies and critics use raw weights. The league used 256 fixed final-stage
competition boards for every matchup, two seat-swapped games per board, and
temperature-1 categorical sampling. Each cell contains 512 games. `Score` is
`(wins + 0.5 * draws) / 512`; `win rate` counts draws as zero.

Labels:

- C1: conv 1,000
- C2: continuation 2,000
- C3: continuation 3,000
- C4: lambda=.97 control 4,000
- C4phi: phi-boost treatment 4,000

## Stochastic league

### Score (%)

| row vs column | C1 | C2 | C3 | C4 | C4phi |
|---|---:|---:|---:|---:|---:|
| C1 | 53.32 | 2.93 | 1.86 | 2.05 | 1.46 |
| C2 | 97.07 | 50.29 | 14.16 | 9.77 | 11.23 |
| C3 | 98.14 | 85.84 | 50.49 | 46.97 | 41.80 |
| C4 | 97.95 | 90.23 | 53.03 | 53.42 | 44.14 |
| C4phi | 98.54 | 88.77 | 58.20 | 55.86 | 48.54 |

### Win rate (%)

| row vs column | C1 | C2 | C3 | C4 | C4phi |
|---|---:|---:|---:|---:|---:|
| C1 | 51.37 | 2.93 | 1.76 | 1.95 | 1.37 |
| C2 | 97.07 | 47.27 | 13.67 | 9.18 | 10.74 |
| C3 | 98.05 | 85.35 | 49.02 | 45.12 | 40.62 |
| C4 | 97.85 | 89.65 | 51.17 | 51.76 | 43.36 |
| C4phi | 98.44 | 88.28 | 57.03 | 55.08 | 47.46 |

### Castles built by the row policy

| row vs column | C1 | C2 | C3 | C4 | C4phi |
|---|---:|---:|---:|---:|---:|
| C1 | 0 | 0 | 0 | 0 | 1 |
| C2 | 0 | 0.5 | 1 | 0 | 1 |
| C3 | 0 | 0 | 0 | 0 | 0 |
| C4 | 1 | 0 | 1 | 2.5 | 1 |
| C4phi | 0 | 0 | 0 | 1 | 0 |

### Games where the row policy built at least one castle

| row vs column | C1 | C2 | C3 | C4 | C4phi |
|---|---:|---:|---:|---:|---:|
| C1 | 0 | 0 | 0 | 0 | 1 |
| C2 | 0 | 0.5 | 1 | 0 | 1 |
| C3 | 0 | 0 | 0 | 0 | 0 |
| C4 | 1 | 0 | 1 | 2.5 | 1 |
| C4phi | 0 | 0 | 0 | 1 | 0 |

Diagonal castle cells average the two policy identities as requested. This is
why the diagonal contains 0.5 and 2.5 and why summing the displayed matrix
does not recover the physical event total. The unaveraged result was 13 builds
in 13 games, or 0.1693% of league games. Legal build-action counts exactly
equaled observed castle-mask constructions in every matchup.

### Per-policy castle exposure

Each policy occupied 3,072 player-game seats across the league.

| policy | builds | build games | build-game rate | games with legal opportunity |
|---|---:|---:|---:|---:|
| C1 | 1 | 1 | 0.0326% | 95.02% |
| C2 | 3 | 3 | 0.0977% | 95.90% |
| C3 | 0 | 0 | 0% | 94.11% |
| C4 | 8 | 8 | 0.2604% | 95.64% |
| C4phi | 1 | 1 | 0.0326% | 96.09% |

### Self-play game length

Mean game length on each diagonal 512-game self-play matchup:

| policy | mean turns |
|---|---:|
| C1 | 455.4 |
| C2 | 598.8 |
| C3 | 562.9 |
| C4 | 553.6 |
| C4phi | 550.6 |

Thus the latest raw 4,000 checkpoints typically run about 551--554 turns in
stochastic self-play; raw 3,000 averaged 562.9 turns.

## Critic analysis

Raw-3,000 categorical self-play supplied a common bank of 2,016 legal-build
opportunities from 1,024 source games, stratified by source game, player, and
200-turn bin. All three atlas runs produced the identical selected-state SHA:
`ca4b7f0ef0b210f3383b687517b9a9c0640eb860d92db4dd160100d540bb291d`.
Each state used 16 paired build/control continuations with common future random
numbers and identical opponent intervention actions.

Of 2,016 states, 1,990 had nonterminal build and control successors available
for critic comparison. Value differences are on the network's [-1, 1] value
scale. Causal effects are game-score differences on [0, 1].

### Common-action critic comparison

All five critics evaluated the exact raw-3,000 intervention successors. This
is the clean apples-to-apples test of critic evolution.

| critic | all legal ΔV | causal-positive ΔV | positive-both-halves ΔV | held-out 8/8 good ΔV | fraction ΔV>0 | Spearman(actual, ΔV) |
|---|---:|---:|---:|---:|---:|---:|
| C1 | -0.4420 | -0.4264 | -0.4604 | -0.4410 | 3.27% | 0.263 |
| C2 | -0.4188 | -0.3796 | -0.3976 | -0.3911 | 3.12% | 0.301 |
| C3 | -0.4434 | -0.4021 | -0.4230 | -0.4189 | 2.11% | 0.288 |
| C4 | -0.3907 | -0.3498 | -0.3661 | -0.3627 | 6.53% | 0.292 |
| C4phi | -0.4068 | -0.3774 | -0.3973 | -0.3905 | 3.32% | 0.280 |

The 663 full-sample causal-positive states had mean actual score effect
+0.2473. The stricter 413 positive-in-both-halves states had mean actual score
effect +0.3411. Despite that, every critic's mean ΔV remained strongly
negative. For held-out good builds, the clustered intervals were:

- C3: -0.4189, 95% CI [-0.4461, -0.3927]
- C4: -0.3627, 95% CI [-0.3867, -0.3388]
- C4phi: -0.3905, 95% CI [-0.4132, -0.3683]

Control 4,000 improved over raw 3,000, but did not approach the correct sign.
On identical states, treatment-minus-control ΔV was -0.0161 overall (95% CI
[-0.0203, -0.0118]) and -0.0313 on stable-good states (95% CI [-0.0399,
-0.0224]). The phi treatment therefore made the critic less favorable to the
same castle successors than the control did.

The modest positive rank correlations show that the critics contain some
relative information about which builds are less bad, but their calibration is
wrong enough that even truly beneficial builds almost always remain below the
non-build successor.

### Matched-policy comparison and policy logits

Here each checkpoint chose its own highest-logit legal build/control actions
and continued with its own stochastic policy.

| policy | actual score Δ | successor ΔV | held-out-good ΔV | median build probability | median build margin | median build rank |
|---|---:|---:|---:|---:|---:|---:|
| C3 | -0.0483 | -0.4433 | -0.4189 | 2.26e-9 | -19.79 | 299.5 |
| C4 | -0.0766 | -0.3921 | -0.3943 | 3.49e-9 | -19.25 | 319.5 |
| C4phi | -0.0699 | -0.4069 | -0.4192 | 2.53e-11 | -24.29 | 464.0 |

The phi-boost treatment's median aggregate build probability is about 89x
lower than raw 3,000 and 138x lower than control 4,000. Its median best legal
build fell another 4.5 logits behind the best non-build action. This is the
opposite of the desired tactical exploration change.

Stable-good raw-3,000 states occurred earlier (median turn 422 versus 569.5),
while behind in army (median -8 versus 0) and land (median -7 versus -2), and
farther from enemy land (median distance 3 versus 2). Their median build
probability was even lower than average: 8.46e-10 versus 2.26e-9. These are
descriptive associations, not a standalone build heuristic, but they identify
useful strata for intervention sampling.

## Recommendation

Do not rely on unmodified self-play to escape this basin. Run a controlled
action-kind exploration experiment:

1. In a small mixture of training actors, intervene only when build is legal.
   Start with a 0.5%--2% build-action probability, stratified to oversample the
   empirically promising earlier/behind/safer states, while retaining broad
   coverage. Select sites from the masked build logits or a randomized
   high-garrison shortlist rather than always one hard-coded location.
2. Record the actual behavior-policy probability of every intervention and use
   the algorithm's proper off-policy correction. Do not overwrite a forced
   action with the model's original log probability and then treat it as an
   on-policy sample.
3. Train an auxiliary build-vs-control value-difference/Q head from paired
   continuation targets like this atlas. If reward shaping remains desirable,
   use a potential-difference term and ablate it separately; avoid a permanent
   unconditional reward for constructing a castle, since most legal builds are
   genuinely harmful.
4. Keep an unmodified control and gate the intervention schedule on held-out
   diagnostics: common-action good-build ΔV, good-state build probability,
   castle frequency, and head-to-head score against the no-build checkpoint.
   Decay forcing only after the policy assigns meaningful probability to the
   good subset and the critic's sign begins to agree with its causal return.

The key target is selective castle competence, not a higher unconditional
build rate. The current treatment did not achieve that target.

## Artifacts

- `stochastic_league.json`: full matrices, W/L/D, paired confidence intervals,
  build opportunities, successful constructions, and seat-attributed counts.
- `critic/*/atlas.json`: per-policy causal summaries and checkpoint hashes.
- `critic/*/paired_rollouts.npz`: compact state features, paired outcomes, and
  critic values; no trajectories.
- `critic_analysis/critic_analysis.json`: matched-policy and common-action
  statistics with source-game-clustered intervals.
- `critic_analysis/01_critic_response_by_checkpoint.png`: critic calibration
  on all and held-out-good builds.
- `critic_analysis/02_treatment_control_paired_shift.png`: paired treatment
  minus control shift on identical successors.
- `critic_analysis/03_causal_vs_critic.png`: actual causal effect versus critic
  ΔV for raw 3,000, control 4,000, and treatment 4,000.
- `critic_analysis/04_matched_policy_groups.png`: matched-policy all/stable/
  held-out-good groups.
- `critic_analysis/05_value_decomposition.png`: build successor versus pre-state
  and versus control successor.
- `critic_analysis/06_policy_critic_alignment.png`: causal effect by critic ΔV
  decile.
- The exact downloaded node export was locally hash-verified as
  `88ca5ca7da41a409d7f1f69f6fb2626083c06a43878027342b3ab74de4f1847a`;
  redundant tar/log exports are intentionally excluded from git.

The experiment ran on one H100. The node `generals-castle-followup` was stopped
after the export hash was verified locally; givemeanode reports `stopped (disk
intact)`.

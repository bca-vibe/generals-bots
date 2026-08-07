# Concurrent PPO + Supervised Castle Counterfactual Training Plan

## Document status

This is an implementation and experiment handoff. It describes the proposed
next castle-learning experiment; none of the changes below have been
implemented yet.

The central idea is to continue ordinary PPO self-play while concurrently
generating paired castle counterfactuals and training auxiliary supervised
losses from them. Counterfactual branches are intervention data, not PPO data.
They must never be inserted into PPO with fabricated on-policy log
probabilities.

## Executive summary

The current policy almost never builds, even though paired rollouts show that
building is strongly beneficial in a learnable subset of ordinary self-play
states. The current critic also predicts a large value decrease after building
in those states. Ordinary PPO therefore has almost no mechanism for crossing
the no-build basin:

- useful builds have negligible sampling probability;
- the policy logits do not rank useful states or sites well;
- terminal-only credit is long-horizon;
- the state-value critic debits the construction cost without adequately
  crediting future castle production;
- most legal builds are harmful, so an unconditional castle reward or blanket
  logit bonus teaches the wrong behavior.

The proposed treatment has two simultaneous data streams:

1. **Ordinary PPO stream:** unchanged symmetric self-play, GAE, PPO clipping,
   entropy, and HL-Gauss value training.
2. **Counterfactual stream:** sample legal-build states, clone each state, force
   selected build and non-build actions, run paired continuations, and train
   supervised actor-preference, successor-value, and build-difference losses.

The desired outcome is selective castle competence: the actor should build in
states where paired continuations say it improves game outcome, avoid building
where it is harmful, and then allow ordinary PPO to refine the behavior using
naturally sampled castles.

## Evidence motivating the experiment

At checkpoint 3,000:

- forced builds everywhere were harmful on average;
- a substantial subset of states had stable positive causal build effects;
- selectors using policy-observable information achieved positive held-out
  uplift;
- aggregate build probability was effectively zero and had essentially no
  correlation with causal value;
- critics predicted strongly negative build-minus-control successor values,
  including on held-out causally beneficial builds.

Relevant reports:

- `runs/castle_counterfactual_atlas_iter3000/README.md`
- `runs/castle_value_probe_iter3000/README.md`
- `runs/castle_raw_checkpoint_followup_20260804/README.md`
- `runs/castle_phi_boost_ab_lambda097_from_003003_20260804/README.md`

The previous potential-shaping/logit-boost treatment should not be reused in
this experiment. Its castle behavior did not persist, and its terminal policy
assigned even less probability to builds.

## Experiment hypothesis

Paired counterfactual supervision can supply a reliable action-value signal in
states where PPO cannot sample a useful build. Interleaving that signal with
ordinary PPO should:

1. teach the actor to rank useful build actions above its normal alternative;
2. teach the critic to value actual post-build states correctly;
3. teach a spatial auxiliary head to predict build-versus-control value;
4. preserve ordinary movement competence through continued on-policy PPO;
5. eventually make natural castle trajectories common enough for PPO to take
   over most of the learning.

## Non-goals and constraints

- Do not maximize unconditional castle frequency.
- Do not add a permanent reward merely for constructing or owning a castle.
- Do not use true hidden enemy geometry as a model input or training feature.
- Do not treat forced counterfactual actions as on-policy PPO samples.
- Do not replace the raw terminal-return target used by the ordinary critic.
- Do not use the old tactical potential or heuristic logit boost in the first
  treatment. Set their scales to zero so the counterfactual intervention is
  isolated.
- Do not require the auxiliary build-difference head at competition inference
  time. The actor must learn the build behavior in its ordinary policy logits.

## High-level data flow

For every training cycle:

1. Collect the normal PPO rollout exactly as today.
2. Identify and reservoir-sample player-steps with at least one legal build.
3. Preserve the raw game state, player seat, observation memory, augmented
   observation, temporal history, and legal mask for selected opportunities.
4. For each selected opportunity, propose one or more legal build actions and
   a non-build control action.
5. Clone the source state into matched build/control branches.
6. Hold the opponent intervention-turn action fixed across each pair.
7. Snapshot the latest completed raw policy, freeze that snapshot for the
   counterfactual refresh, and continue both branches with it and common random
   keys until termination or truncation.
8. Convert paired terminal outcomes into supervised targets.
9. Add the examples to a bounded, checkpoint-versioned replay buffer.
10. During PPO optimization, mix small counterfactual minibatches into the
    total loss without changing PPO ratios or advantages.

“Concurrent” means the losses and data distributions overlap during the same
training run. The counterfactual rollout generation may initially be
synchronous and periodic for simplicity. It can later become an asynchronous
sidecar on a spare accelerator if its measured overhead is material.

## Counterfactual opportunity sampling

### Eligibility

A player-step is eligible when at least one castle build is legal under the
canonical action mask. Selection must be based only on information available
to the policy, although the raw environment state must be retained internally
so the simulator can fork the game correctly.

### Exact source-opportunity sampler

Across the 20 PPO iterations between refreshes, stream all eligible steps into
grouped reservoirs. The grouping key is:

```text
episode_id, player_seat, floor(turn / 200)
```

Retain at most one uniformly reservoir-sampled opportunity per group. This
prevents long games, one player seat, or repeated affordable-build turns from
dominating. At refresh time, select exactly 256 source opportunities from the
group winners using these quotas:

| Source quota | Count | Selection rule |
|---|---:|---|
| Uniform coverage | 128 | Uniform across eligible group winners |
| Promising | 64 | Uniform among observable earlier/behind/adequate-garrison/safer candidates |
| Hard/informative | 64 | 32 highest current build margins plus 32 feature-diverse candidates |

The promising gate may use turn, army and land margins, exact build cost,
post-build garrison, structure distance, and visible or remembered enemy
pressure. It must not use true hidden enemy geometry.

For the first experiment, feature-diverse hard examples should be selected by
balanced bins over turn, army margin, land margin, cost, and post-build
garrison. Do not make source selection depend on a counterfactual label that
does not exist yet. The auxiliary build-difference head may be evaluated as a
diagnostic, but it does not alter this fixed 400-iteration source sampler.

Deduplicate states selected through multiple quotas, prioritize distinct source
games, and backfill any shortfall from the uniform group-winner pool. Record
the group counts, quota, selection probability, and observable features for
population-weighted diagnostics. The promising quota is a compute curriculum,
not a positive label; harmful examples from every quota remain in training.

### Initial recommended scale

Use the following concrete setting for the first throughput preflight and A/B:

- 256 new source opportunities per refresh across the whole run;
- refresh every 20 PPO iterations;
- 2 build candidates per opportunity;
- 4 paired repetitions per candidate during training;
- a larger held-out evaluation with 8-16 repetitions;
- rolling buffer capacity of 30,000 state/candidate examples;
- expire or strongly downweight examples generated more than 100 PPO
  iterations ago.

This produces `256 * 2 * 4 = 2,048` paired continuations, or 4,096 total branch
rollouts, per refresh. Increase to 512 source opportunities only after the
preflight shows that generation is asynchronous or remains inside the measured
overhead budget.

The effective target is policy-dependent, so stale examples must not dominate.

## Action proposal and paired branch protocol

### Build candidates

Do not evaluate only the actor's current favorite build site. The existing
policy has poor build-site ordering, and doing so would make the dataset inherit
that failure.

For every source opportunity, candidate 1 is always the highest-logit legal
build. Candidate 2 is selected reproducibly from a three-way rotating proposal:

1. highest post-build-garrison legal build;
2. randomized legal build, weighted toward adequate post-build garrison;
3. closest legal build to the agent's general by Manhattan distance.

Choose the second proposal with a stable hash of source game, player seat,
turn bin, and raw generator iteration modulo three. Use deterministic row-major
tie-breaking for equally close general-distance cells. Deduplicate candidate 2
against candidate 1; on collision, try the next proposal in the rotation. If
only one distinct legal build exists, emit one state/candidate example rather
than duplicating it.

Record the exact proposal method, legal candidate count, and proposal
propensity. Do not use the auxiliary head to propose sites during this first
400-iteration experiment.

### Control action

For the first implementation, use the actor's highest-logit legal non-build
action, matching the established atlas. This gives a concrete, conservative
pairwise preference target and directly attacks the current build rank.

A later ablation may use an action drawn from the policy conditioned on
non-build actions. That estimates build versus the policy's expected ordinary
alternative rather than versus one specific control.

### Matched continuation

For every source state, build candidate, and repetition:

1. At refresh start, copy the latest completed raw-policy parameters and freeze
   that version for the entire paired batch.
2. Sample one opponent intervention-turn action from that frozen raw snapshot.
3. Use exactly that opponent action in both branches.
4. Force the actor to take the build in one branch and the control in the
   other.
5. Preserve the correct recurrent observation memory for both seats.
6. Continue both seats using the same frozen raw-policy snapshot.
7. Reuse corresponding categorical random keys across branches at every future
   step.
8. Run to terminal win/loss or the ordinary 1,200-turn draw truncation.

Common keys do not make the branches identical after they diverge, but they
substantially reduce paired variance. Store the frozen raw snapshot's training
iteration, checkpoint ID, hash, and generation lag with every example.

## Targets

Represent terminal outcome on the critic's existing scale:

- win: `+1`
- draw: `0`
- loss: `-1`

This is confirmed by the training implementation, not merely a proposed
convention. The environment emits terminal rewards `+1` and `-1` to the winner
and loser respectively, and `0` for ongoing steps and a truncation draw. A full
Monte Carlo continuation therefore has outcome exactly in `{-1, 0, +1}` with
`gamma = 1`, and those are the targets for the supervised counterfactual
successor loss. The value head is correspondingly configured on `[-1, 1]`.

Ordinary PPO critic targets are not restricted to those three discrete numbers:
the trainer uses GAE with `lambda = 0.90` and forms a bootstrapped lambda-return,
so intermediate PPO value targets are continuous on the same `[-1, 1]`
win/draw/loss semantic scale. They are still not on the evaluator's
`{0, 0.5, 1}` score scale.

Evaluation reports and the existing counterfactual atlas instead use the game-
score scale `{0, 0.5, 1}` for loss/draw/win. Any training data derived from
those evaluator outcomes must be converted before constructing critic or
delta targets:

```text
critic_outcome = 2 * evaluation_score - 1
critic_delta = critic_outcome_build - critic_outcome_control
             = 2 * (score_build - score_control)
```

Prefer constructing the training target directly from the terminal winner and
truncation flags so this conversion cannot be omitted accidentally.

For source state `s`, build `b`, control `c`, and repetition `k`, record:

```text
z_build[k]   = terminal outcome after forcing b
z_control[k] = terminal outcome after forcing c
delta[k]     = z_build[k] - z_control[k]
```

The state/candidate target is:

```text
delta_mean = mean(delta[k])
delta_se   = standard_error(delta[k])
```

The delta scale is `[-2, 2]`. Do not accidentally use the atlas's `[0, 1]`
score scale without converting it.

For noisy actor labels, shrink the effect toward zero:

```text
delta_shrunk = sign(delta_mean) * max(abs(delta_mean) - kappa * delta_se, 0)
```

Start with `kappa = 1`. Examples whose shrunk effect is zero should have zero
or very low actor-preference weight, while their individual successor outcomes
can still train the value head.

## Model changes

### Existing heads

The ordinary spatial policy head and scalar HL-Gauss value head remain the
deployment actor and critic. Supervised gradients should update them and the
shared representation, subject to conservative loss weights.

### New spatial build-difference head

The spatial build-difference head is included in the first 400-iteration
treatment and is not an optional ablation. Add a training auxiliary head
producing one scalar for each board cell:

```text
D(s, b) ~= Q^pi(s, build_at_b) - Q^pi(s, control)
```

Suggested implementation:

- apply a linear projection to each final spatial patch token;
- emit `patch_size * patch_size` values per patch;
- reshape them to a 21x21 cell grid;
- mask illegal build cells for metrics and sampling;
- zero-initialize the final projection so attaching the head does not change
  the loaded actor or critic outputs.

The target range is `[-2, 2]`. A scalar regression head with Huber loss is
sufficient; it does not need the critic's `[-1, 1]` HL-Gauss bins.

### Checkpoint compatibility

Adding Equinox leaves will make naive deserialization of historical
checkpoints fail. Implement an explicit versioned migration:

1. instantiate and deserialize the historical actor-critic architecture;
2. wrap or upgrade it with the zero-initialized auxiliary head;
3. save new checkpoints with a schema/version marker;
4. ensure competition export can omit or ignore the training-only head;
5. add a regression test showing that attaching the head changes ordinary
   logits and scalar values by exactly zero before training.

Avoid silently changing the historical checkpoint tree definition.

## Supervised losses

### 1. Actor pairwise preference loss

For pre-action logits `l_b` and `l_c`, define the model's conditional
preference for the sampled pair:

```text
p_model = softmax([l_b, l_c])[build]
p_target = sigmoid(delta_shrunk / actor_target_temperature)
```

Then:

```text
L_actor_cf = weight * binary_cross_entropy(p_model, p_target)
```

Suggested weight:

```text
weight = clip(abs(delta_shrunk) / actor_weight_scale, 0, 1)
```

This is advantage-weighted preference learning rather than hard behavioral
cloning. Positive examples raise the selected build relative to the action the
actor would otherwise take; negative examples lower it. Sampling multiple
build sites teaches site choice as well as action kind.

Initial suggested values:

- `actor_target_temperature = 0.25` on the `[-2, 2]` delta scale;
- `actor_weight_scale = 0.5`;
- exclude zero-weight examples from actor metrics.

### 2. Successor-state value loss

The current critic is `V(s)`, not `Q(s, a)`. Do not assign both branch outcomes
to the same pre-action state value.

Instead, train the existing value head on the distinct post-intervention
successor inputs:

```text
V(successor_build)   -> z_build
V(successor_control) -> z_control
```

Use the existing HL-Gauss cross-entropy with the ordinary value bins and sigma.
Each repetition may have a different paired opponent action and therefore a
different successor state; retain the matching successor input and outcome.

This is the most direct correction for the observed critic failure: it teaches
the scalar value head that a state containing a newly built castle can be good
despite the immediate army debit.

### 3. Build-difference loss

Train the new spatial head at the selected build cell:

```text
L_delta_cf = huber(D(s, b) - delta_mean)
```

Optionally weight it by inverse target variance with a conservative cap. Do
not allow a few low-variance or repeated states to dominate.

The auxiliary head serves three purposes:

- supplies a direct action-conditioned critic signal at the pre-action state;
- improves the shared representation of castle economics and safety;
- provides uncertainty/disagreement signals for future counterfactual
  sampling.

## Combined optimization

The ordinary PPO loss remains:

```text
L_ppo = L_policy_ppo + value_coefficient * L_value_ppo
        - entropy_coefficient * entropy
```

For update steps with a counterfactual minibatch:

```text
L_total = L_ppo
          + lambda_actor_cf * L_actor_cf
          + lambda_value_cf * L_successor_value_cf
          + lambda_delta_cf * L_delta_cf
```

Starting coefficients for a preflight, subject to gradient-norm calibration:

```text
lambda_actor_cf = 0.05
lambda_value_cf = 0.05
lambda_delta_cf = 0.05
```

The numbers themselves are less important than controlling gradient scale.
Log the gradient norm from each loss separately. Initially target total
counterfactual gradient norm at roughly 10-25% of the PPO gradient norm. Reduce
individual coefficients if the auxiliary stream dominates the shared trunk.

Sample a small counterfactual minibatch independently of the PPO top-advantage
indices. The current top-25% advantage filter must not decide whether a
counterfactual example is retained.

### Exact optimizer minibatch composition

Use counterfactual supervision on every ordinary PPO optimizer step. With the
current `minibatch_size = 1024`, each accelerator shard should receive:

| Example stream | Per device | Global on 4 devices |
|---|---:|---:|
| Ordinary PPO examples | 1,024 | 4,096 |
| Actor counterfactual state/candidate pairs | 64 | 256 |
| Critic successor-value examples | 64 | 256 |
| Total neural examples | 1,152 | 4,608 |

The actor and successor minibatches are sampled independently from two flat
buffer views:

- the actor view contains one pre-action state, build action, control action,
  and paired delta target per state/candidate example;
- the successor view contains individual build or control successor inputs and
  their terminal outcome targets, flattened across branch and repetition.

Construct each global four-device actor batch centrally, then shard it evenly
across devices. The 256 global actor examples use these fixed quotas:

| Actor quota | Global | Per device after sharding |
|---|---:|---:|
| Uniform recent-buffer examples | 128 | 32 |
| Confident positive (`delta_shrunk > 0`) | 64 | 16 |
| Confident negative (`delta_shrunk < 0`) | 64 | 16 |

Within each quota, draw 75% from the latest two completed refreshes and 25%
from older examples that remain within the maximum age. Prefer distinct source
games and do not include the same state/candidate record twice in one global
batch when enough records exist. The uniform quota includes uncertain examples.

Because the positive and negative quotas intentionally alter label prevalence,
store each example's total minibatch selection probability and apply a
normalized, conservatively clipped inverse-sampling correction before the
existing magnitude/confidence loss weight. This balancing must not teach an
artificial 50/50 prior for building. If a sign bucket is short during startup,
backfill from the uniform pool rather than repeatedly duplicating its few
members.

Construct the global critic successor batch independently. Sample 128 recent
`(state, candidate, repetition)` pair IDs without outcome stratification and
include both the build and control successor for each ID. This yields 256
successor records globally, 64 per device, with exact 50/50 branch balance.
Do not balance by terminal win/loss/draw because that would distort value
calibration.

Do **not** select 64 actor examples and then evaluate every one of their
`2 * repetitions` successors on that optimizer step. At four repetitions that
would add 512 successor forwards per device and make the update much more
expensive. Preserve all successors in the buffer, but sample only 64 flattened
successor records per device per optimizer step.

There are 128 optimizer minibatches per iteration under the current
configuration. Each device therefore processes, per iteration:

- 131,072 PPO examples;
- 8,192 actor counterfactual examples;
- 8,192 successor-value examples.

Across four devices, the actor sees 32,768 counterfactual presentations and
the critic sees 32,768 successor presentations per PPO iteration. These are
replay presentations and need not be unique.

Use one optimizer and a single combined gradient for the first implementation.
Keep the optimizer schedule and EMA update count identical between control and
treatment: one combined optimizer step and one EMA update for each ordinary PPO
minibatch.

### Why 64 examples per device is the recommended start

Sixty-four is sufficient as an initial per-device batch because it becomes 256
actor examples globally after data-parallel aggregation, and it is presented
on all 128 optimizer steps. The counterfactual labels are also much denser than
ordinary PPO samples: every retained actor example was deliberately selected,
paired to a control, rolled to terminal outcome, uncertainty-shrunk, and
magnitude-weighted.

If losses are averaged over their minibatches, increasing from 64 to 128 does
not double the expected counterfactual gradient. It mainly reduces gradient
variance, increases source-state coverage per step, and doubles that stream's
neural compute. Adjust the loss coefficient when the mean supervised gradient
is too weak; increase the minibatch size only when diagnostics show excessive
variance or poor coverage.

Construct global batches so confident positive and negative labels and multiple
source games are represented using the central sampler above rather than
independent uncoordinated shard-local draws.

During buffer cold start, scale all configured counterfactual loss coefficients
by:

```text
min(1, unique_state_candidate_examples / 2048)
```

Log both configured and effective coefficients. This prevents the first
approximately 512 labels from being replayed at full strength hundreds of
times before the buffer contains enough distinct source games. The ramp changes
loss strength only; the 64-per-device tensor shapes remain fixed after the
counterfactual path is enabled.

Revisit the batch size after the smoke A/B. Increase actor examples to 128 per
device only if at least one of the following holds despite a healthy buffer:

- supervised gradient norms or held-out margins are highly variable between
  optimizer steps;
- too few confident positive examples occur in typical global minibatches;
- actor ordering improves on replayed examples but is unstable on held-out
  source games;
- accelerator profiling shows that 128 examples add negligible marginal time.

## Strict PPO accounting boundary

Counterfactual branches are supervised intervention samples. For them:

- do not calculate a PPO probability ratio;
- do not use the source policy's original log probability for a forced action;
- do not pass them through GAE;
- do not place them in the ordinary rollout tensor as if the behavior policy
  selected the action;
- do not include them in PPO KL or clip-fraction metrics.

The normal PPO stream should continue to store and recompute the exact behavior
policy log probability exactly as it does now.

No explicit build-action probability floor is included in the first A/B. The
supervised actor loss itself should raise useful build logits. If it does not
produce natural builds, run a separate follow-up with a correctly accounted
0.5%-2% action-kind exploration mixture rather than changing this experiment
mid-run.

## Replay buffer contents

The existing atlas NPZ files contain compact features and outcomes but not the
full neural inputs required for end-to-end training. Generate a new buffer
format containing at least:

### Pre-action fields

- augmented observation, preferably bfloat16;
- temporal history, preferably bfloat16;
- legal action mask;
- player seat;
- build action index;
- control action index;
- source game/map identifier;
- source turn and sampling stratum;
- proposal and selection propensities;
- generator checkpoint iteration and hash.

### Label fields

- per-repetition build and control outcomes;
- `delta_mean`, `delta_se`, and `delta_shrunk`;
- completion/truncation flags;
- optional survival/payback diagnostics.

### Successor-value fields

- augmented build successor observation/history/mask per repetition;
- augmented control successor observation/history/mask per repetition;
- matching terminal outcome target for each successor;
- whether the intervention transition itself terminated the game.

Use bounded shards with an atomic completed-file rename if a sidecar process
writes them. The trainer must ignore partial shards and incompatible schema
versions.

## Scheduling and policy freshness

Generate labels with a frozen snapshot of the latest completed **raw** policy.
Do not use EMA as the primary online counterfactual continuation policy, and do
not allow the raw snapshot to change inside one paired batch.

The target should approximate
`Q^(current raw policy)(state, build) - Q^(current raw policy)(state, control)`
because the raw policy generates the ordinary PPO distribution and receives
the supervised update. With the current once-per-iteration EMA decay of 0.999,
EMA labels would remain strongly tied to the original policy during the
200-300-iteration pilot and could become stale as post-build behavior changes.

For an asynchronous sidecar, "latest raw" means the newest atomically completed
raw checkpoint it can load. A lag of one refresh is acceptable if it is
recorded, bounded, and included in freshness metrics.

Suggested freshness rules:

- publish or copy a new frozen raw generator snapshot every 20 PPO iterations;
- tag every buffer item with the raw generator iteration and hash;
- log current-training-iteration minus generator-iteration lag;
- sample recent examples preferentially;
- expire examples beyond a fixed maximum age;
- periodically retain a small fixed diagnostic set, but do not train on it.

This keeps the supervised target close to `Q` under the policy currently being
optimized while avoiding within-pair nonstationarity.

EMA remains useful for secondary deployment-oriented evaluation and a possible
raw-versus-EMA continuation ablation. It is not the source of primary training
labels.

## Implementation map

Expected primary touchpoints:

- `generals/training/model.py`
  - expose final token representations cleanly;
  - add or support the spatial build-difference head;
  - preserve ordinary forward compatibility.
- `generals/training/conv_model.py`
  - route the convolutional competition model through the same auxiliary head
    path without changing deployment outputs.
- `generals/training/ppo.py`
  - keep PPO math unchanged;
  - add an optional counterfactual minibatch and the three supervised losses,
    or create a separate counterfactual epoch function.
- `generals/training/train.py`
  - capture opportunity states;
  - manage refresh cadence and replay buffer sampling;
  - combine/log losses and gradient norms;
  - checkpoint the auxiliary head and buffer metadata.
- `generals/training/rollout.py`
  - expose raw source states and per-seat memory for selected opportunities
    without copying every rollout state to the host.
- `generals/training/config.py`
  - add explicitly validated counterfactual settings.
- `generals/training/tracking.py`
  - route counterfactual metrics into separate namespaces.
- `tools/evaluate_castle_counterfactuals.py`
  - refactor the reusable paired-continuation kernel into a training module
    rather than duplicating its branch logic.

Prefer introducing a focused module such as
`generals/training/counterfactual.py` for opportunity reservoirs, paired
continuations, target construction, and buffer schemas.

## Configuration fields

Suggested fields, with treatment defaults shown illustratively:

```toml
counterfactual_castle_training = true
counterfactual_refresh_every = 20
counterfactual_source_states = 256
counterfactual_build_candidates = 2
counterfactual_repetitions = 4
counterfactual_buffer_capacity = 30000
counterfactual_max_age_iterations = 100
counterfactual_actor_minibatch_size_per_device = 64
counterfactual_successor_minibatch_size_per_device = 64
counterfactual_update_every_minibatches = 1
counterfactual_unique_examples_full_weight = 2048

counterfactual_uniform_source_states = 128
counterfactual_promising_source_states = 64
counterfactual_hard_source_states = 64
counterfactual_actor_uniform_per_device = 32
counterfactual_actor_positive_per_device = 16
counterfactual_actor_negative_per_device = 16
counterfactual_recent_fraction = 0.75

counterfactual_actor_coefficient = 0.05
counterfactual_value_coefficient = 0.05
counterfactual_delta_coefficient = 0.05
counterfactual_actor_temperature = 0.25
counterfactual_actor_weight_scale = 0.5
counterfactual_uncertainty_kappa = 1.0
counterfactual_huber_delta = 0.25

# Explicitly disabled in the first experiment.
actor_potential_shaping = false
castle_potential_scale = 0.0
tactical_build_logit_boost = 0.0

# The implementation may require an absolute endpoint rather than a
# continuation count. Set this to resume_iteration + 400 when appropriate.
num_iterations = <resume_iteration + 400>
eval_every = 50
league_eval_every = 50
checkpoint_every = 100
latest_checkpoint_every = 100
```

All new settings must validate cleanly, and setting
`counterfactual_castle_training = false` must follow the baseline path without
allocating the buffer or compiling auxiliary kernels.

## Metrics

### Natural castle use in stochastic PPO rollouts

Log castle behavior from the ordinary stochastic self-play trajectories used
to train PPO. These metrics must use the actions actually sampled and executed
by the behavior policy. Do not include forced counterfactual branches, replay
examples, deterministic evaluation, or actions proposed but not executed.

Record the following for every PPO iteration, for both control and treatment:

- `ppo_castle_build_move_rate`: number of executed legal castle-build actions
  divided by the number of policy action decisions;
- `ppo_castle_build_game_rate`: number of completed self-play games in which at
  least one player executed a castle build divided by completed self-play
  games;
- `ppo_castle_build_player_game_rate`: number of completed episode-seat pairs
  in which that seat executed at least one castle build divided by completed
  episode-seat pairs;
- `ppo_castle_builds_per_game`: total executed castle builds divided by
  completed self-play games;
- `ppo_castle_build_eligible_rate`, as a diagnostic: executed castle builds
  divided by policy decisions at which at least one castle build was legal.

Always log the raw numerator and denominator beside each rate. Report both the
per-iteration values and an aggregate over each 50-iteration evaluation
window. Track game-level flags by episode ID across rollout-fragment and PPO
iteration boundaries, and count a game exactly once when the environment ends
it. Report terminated and time-limit-truncated games separately so a change in
episode length or truncation rate cannot masquerade as a change in castle use.
The primary metrics above pool both self-play seats because both use the same
behavior policy; retain per-seat breakdowns as a symmetry/debugging check.

### Data generation

- eligible opportunities observed;
- opportunities sampled by stratum and turn bin;
- candidate proposal counts;
- paired continuations completed;
- mean and distribution of `delta_mean`;
- positive, negative, and uncertain label fractions;
- mean target standard error;
- buffer size, age distribution, and generator checkpoint lag;
- raw generator iteration/hash and current-minus-generator iteration lag;
- counterfactual generation wall time and accelerator overhead.

### Actor

- pairwise preference loss;
- build-minus-control logit margin on positive examples;
- build-minus-control logit margin on negative examples;
- aggregate legal build probability by causal label;
- best-build rank by causal label;
- rank correlation between actor margin and held-out causal delta;
- the natural PPO-rollout castle-use metrics defined above;
- natural build raw advantage and retention through the top-advantage filter.

### Critic

- successor-value supervised loss;
- predicted `V(build successor) - V(control successor)` by causal group;
- fraction of held-out positive builds whose predicted successor delta is
  positive;
- build-difference Huber loss;
- build-difference sign accuracy and Spearman correlation with held-out delta;
- ordinary PPO explained variance, to detect value degradation.

### Optimization and safety

- PPO-only gradient norm;
- actor-counterfactual gradient norm;
- successor-value gradient norm;
- build-difference gradient norm;
- cosine similarity between PPO and each auxiliary gradient if affordable in
  preflight diagnostics;
- ordinary approximate KL and clip fraction;
- iteration time, samples/second, and counterfactual overhead.

## Evaluation protocol

Run a matched A/B from the same checkpoint:

Run both branches for exactly 400 continuation iterations unless a correctness
or numerical-stability failure invalidates the run. Evaluate both branches at
continuation iterations 50, 100, 150, 200, 250, 300, 350, and 400. Save numbered
raw, EMA, optimizer, and training-state checkpoints for both branches at
continuation iterations 100, 200, 300, and 400. A replace-in-place latest
checkpoint may also be written every 100 iterations; it does not replace the
numbered artifacts.

### Control

- ordinary PPO continuation;
- counterfactual training disabled;
- potential shaping disabled;
- tactical build boost disabled.

### Treatment

- identical PPO configuration, seeds, curriculum stage, hardware allocation,
  and wall-clock or sample budget;
- concurrent supervised counterfactual training enabled;
- no other castle intervention.

If counterfactual generation consumes meaningful extra accelerator time,
report both allocated GPU-hours and ordinary PPO samples. Prefer equal
allocated compute for the main comparison, with an additional sample-matched
interpretation if useful.

### Periodic diagnostics

Every 50 continuation iterations, evaluate both raw and EMA policies on:

1. ordinary paired-map league against the frozen source checkpoint and control;
2. stochastic self-play castle frequency and legal-opportunity exposure;
3. a fixed, never-trained-on counterfactual diagnostic bank;
4. a small fresh counterfactual sample to detect diagnostic-set overfitting;
5. common-action successor critic comparisons using identical source states.

At the same cadence, publish the preceding 50 iterations of stochastic PPO
rollout castle-use metrics for both branches. Keep these on-policy behavior
statistics separate from the evaluation-game castle frequencies above.

### Terminal evaluation

After continuation iteration 400:

- raw and EMA round robin among source, control, and treatment;
- at least 4,096 stochastic games per key self-play or matchup castle audit;
- a fresh atlas grouped by source game with 8-16 paired repetitions;
- actor/critic causal calibration plots;
- population and promising-stratum results;
- castle survival, payback, and repeated-castle behavior;
- bootstrap confidence intervals clustered by source game/map.

## Pilot timeline and decision schedule

The pilot iteration clock starts when the first usable counterfactual buffer is
available and supervised losses are enabled, not necessarily at continuation
iteration 1. Log `counterfactual_ready_iteration`, refresh count, unique source
games, unique state/candidate labels, and cumulative counterfactual examples
consumed so every checkpoint can be interpreted correctly.

At the expected synchronous treatment throughput of roughly 275-305 iterations
per hour, excluding JIT compilation and evaluation pauses:

- 50 enabled iterations take approximately 10-11 minutes;
- 100 enabled iterations take approximately 20-22 minutes;
- 400 enabled iterations take approximately 79-87 minutes.

The experiment should yield a mechanistic answer within 50-100 enabled
iterations, but the planned A/B runs for the full 400 iterations so later PPO
adaptation and retention can be observed. Intermediate diagnostic failures do
not change the intervention or stop a valid run; they are recorded for the
terminal interpretation.

### Iterations 1-5: implementation sanity

The dense supervised losses should react almost immediately. Confirm:

- actor and successor-value counterfactual losses are finite;
- positive labels raise build-versus-control margins;
- negative labels lower build-versus-control margins;
- successor-value error begins decreasing;
- all masks, target scales, and branch labels pass runtime assertions;
- PPO KL, entropy, explained variance, and gradient norms remain stable;
- total counterfactual gradient norm is approximately 10%-25% of PPO gradient
  norm, or is deliberately moving toward that calibrated range.

If supervised metrics do not move after five enabled iterations, treat it as an
implementation, masking, coefficient, or optimizer-integration problem rather
than evidence against the learning hypothesis.

### Iterations 20-50: held-out mechanistic evidence

Evaluate the raw network on the fixed, never-trained-on diagnostic bank. Do not
accept falling replay loss as sufficient evidence.

The actor should show:

- increasing build-minus-control margin on causally positive examples;
- little increase or a decrease on negative examples;
- growing positive-versus-negative margin separation;
- improving Spearman correlation between build margin and paired causal delta;
- improving build rank on beneficial examples.

The critic should show:

- `V(build successor) - V(control successor)` becoming less negative on
  beneficial examples;
- greater improvement on positive examples than on negative examples;
- improving correlation between predicted and actual paired effects.

Suggested early thresholds by iteration 50:

- actor margin/causal-effect Spearman increases from approximately 0.02 to at
  least 0.10;
- positive-versus-negative mean margin separation improves by at least one
  logit relative to the source checkpoint and matched control;
- mean successor-value difference on held-out good builds improves by at least
  0.10 from its current approximately `-0.42` baseline.

The critic need not cross zero by iteration 50. If replay losses improve but
held-out metrics do not, suspect memorization, insufficient source-game
diversity, stale labels, or an overly narrow candidate sampler.

### Iterations 75-100: learning-mechanism go/no-go

With the recommended cadence, five refreshes provide approximately 1,280 new
source opportunities and up to 2,560 state/build-candidate labels. By this
point require:

- clearly positive held-out actor ordering correlation, preferably at least
  0.15-0.20;
- substantially higher build probability on beneficial states than harmful
  states;
- orders-of-magnitude improvement from the approximately `1e-9` good-state
  build-probability baseline, even if absolute probability remains modest;
- good-build successor-value differences approaching zero;
- some unforced castles in raw-policy self-play;
- no material degradation in PPO explained variance, movement behavior, or
  ordinary playing strength.

If neither held-out actor ordering nor critic calibration has materially
improved by iteration 100, mark the learning-mechanism gate as failed but
continue the unchanged treatment through iteration 400 unless a correctness or
numerical-stability problem invalidates the run.

### Iterations 150-200: natural-policy evidence

The supervised behavior should now be entering ordinary PPO rollouts. Require:

- natural, unforced castles occurring often enough to measure;
- neutral or positive mean raw PPO advantage for natural builds in useful
  strata;
- adequate retention of natural builds through the absolute-advantage filter;
- selective rather than indiscriminate growth in build probability;
- fresh counterfactuals from the current raw policy showing positive causal
  value where its build probability is highest.

If actor and critic held-out diagnostics improve but natural builds remain
absent by iteration 150-200, the pairwise loss may have improved relative
ordering without overcoming the absolute action-probability deficit. Flag that
interpretation, but continue the unchanged treatment through iteration 400. Do
not silently add an action-kind calibration or exploration floor mid-run.

### Iterations 200-400: strategic evidence and terminal verdict

Continue scheduled evaluations at 200, 250, 300, 350, and 400. At iteration
400, run the full fresh held-out atlas, raw-policy head-to-head evaluation,
common-action critic probe, and negative-state false-positive analysis. The
treatment is strategically successful only if:

1. states receiving high build probability have positive fresh paired effects;
2. harmful states remain predominantly no-build;
3. critic successor ordering agrees with causal outcomes;
4. natural builds receive favorable PPO credit;
5. overall playing strength is preserved and preferably improved.

Castle count alone is not evidence of success.

### Raw versus EMA policy during the pilot

Use the raw policy for all early 5/20/50/100/200-iteration decisions. The
current EMA is updated only once per full PPO iteration with decay 0.999. Its
approximate contribution from new parameters is therefore:

- 4.9% after 50 iterations;
- 9.5% after 100 iterations;
- 25.9% after 300 iterations;
- 33.0% after 400 iterations;
- a half-life of approximately 693 iterations.

EMA-only evaluation would make a working treatment appear inactive during the
pilot. Retain EMA evaluation for deployment-oriented confirmation, but do not
use it as the early stop signal. Add a raw-policy diagnostic path if the
existing evaluator only reports EMA.

### Predeclared continuation and abort rules

- Run both valid branches for 400 continuation iterations.
- Evaluate every 50 iterations and checkpoint both branches every 100.
- Do not tune coefficients, sampling, candidates, or exploration mid-run in
  response to an unfavorable intermediate result.
- Abort and restart only for an implementation error, corrupted/missing paired
  labels, non-finite optimization, broken behavior-policy accounting, or a
  comparable condition that invalidates the experiment.
- Make the strategic go/no-go decision after the iteration-400 terminal
  evaluation. Earlier checkpoints diagnose when learning began and whether it
  persisted.

## Success gates

The treatment succeeds only if castle behavior becomes selective and improves
play. Suggested gates:

1. **Actor ordering:** held-out causal delta has a clearly positive correlation
   with build logit margin, materially above the current near-zero result.
2. **Good-state probability:** aggregate build probability rises substantially
   on held-out positive states without a comparable rise on negative states.
3. **Critic sign:** mean predicted build-minus-control successor value moves
   toward or above zero on held-out beneficial builds.
4. **Auxiliary critic:** the build-difference head has positive held-out rank
   correlation and useful sign discrimination on fresh source games.
5. **Natural behavior:** unforced PPO rollouts contain enough castles to
   generate genuine on-policy learning signal.
6. **Selectivity:** forced or natural builds in causally negative states do not
   rise indiscriminately.
7. **Playing strength:** treatment does not regress against control and ideally
   wins a statistically credible head-to-head advantage.
8. **Ordinary competence:** movement metrics, PPO explained variance, and
   league performance show no material degradation attributable to auxiliary
   training.

Castle count alone is not a success metric.

## Annealing criteria

Do not anneal counterfactual supervision during this fixed 400-iteration A/B.
Keep configured coefficients and the unique-example cold-start ramp identical
to the frozen treatment config. For a later continuation experiment, begin
reducing coefficients only when all of the following persist across multiple
evaluations:

- natural builds occur regularly in ordinary PPO rollouts;
- good-state build probabilities remain separated from negative-state
  probabilities;
- natural builds have non-negative mean raw PPO advantage;
- successor critic ordering is no longer strongly wrong on held-out good
  builds;
- removing part of the auxiliary weight does not immediately collapse build
  probability.

Anneal actor supervision first if PPO is clearly sustaining the policy. Retain
the build-difference head longer as a diagnostic and representation loss if it
continues to improve held-out calibration without harming PPO.

## Required tests and preflight checks

### Unit tests

- candidate builders always return legal actions;
- grouped opportunity reservoirs retain at most one state per episode, seat,
  and 200-turn window and reproduce the 128/64/64 source quotas;
- candidate 1 is highest-logit, candidate 2 follows the stable three-way hash
  rotation, collisions backfill deterministically, and closest-to-general
  tie-breaking is row-major;
- control action is legal and non-build;
- paired branches use the identical opponent intervention action;
- outcome and delta scales are correct, including exact conversion of evaluator
  scores `{0, 0.5, 1}` to critic targets `{-1, 0, +1}`;
- actor preference gradients raise build logits for positive labels and lower
  them for negative labels;
- zero-weight uncertain examples do not update the actor through that loss;
- successor value examples never assign two targets to the same pre-action
  `V(s)` prediction;
- build-difference loss indexes the correct spatial cell;
- illegal cells cannot enter actor or auxiliary metrics;
- buffer aging, capacity, schema checks, and atomic shard loading work;
- historical checkpoint migration preserves logits and values exactly before
  training;
- competition export ignores the auxiliary head cleanly.

### Integration tests

- setting all counterfactual coefficients to zero matches baseline PPO metrics
  and updates within numerical tolerance;
- a tiny deterministic paired game produces the expected causal label;
- one multi-device training iteration with counterfactual examples compiles and
  checkpoints;
- resuming the new checkpoint restores actor, critic, auxiliary head,
  optimizer, EMA, and counters;
- forced samples never appear in PPO action/log-probability tensors;
- counterfactual examples bypass the PPO top-advantage filter;
- paired continuations use the immutable raw-policy snapshot identified by the
  stored iteration/hash and never read changing live or EMA parameters;
- each optimizer step samples exactly 64 actor pairs and 64 flattened
  successor records per device rather than expanding all repetitions of the
  selected actor pairs;
- global actor batches reproduce the 128/64/64 uniform/positive/negative
  quotas, record correct selection probabilities, and apply the configured
  cold-start coefficient ramp;
- global successor batches contain both branches of 128 independently sampled
  pair IDs without outcome stratification;
- memory is reset and advanced correctly across source, intervention, and
  continuation states.

### Performance preflight

Before launching the full A/B, measure:

- PPO iteration time with counterfactual training disabled;
- auxiliary-loss-only overhead using a full buffer;
- paired rollout throughput at candidate refresh sizes;
- peak device and host memory;
- checkpoint size and save/load time;
- whether synchronous generation exceeds the acceptable overhead budget.

If generation is too expensive on the training path, move only generation to
an asynchronous sidecar. Keep the trainer's supervised buffer consumption and
losses concurrent with PPO.

### Expected throughput budget

Treat optimizer overhead and paired-continuation generation as separate costs.

The exact optimizer composition above adds 128 neural examples per device to a
1,024-example PPO minibatch, a 12.5% increase in update examples. Because the
ordinary rollout phase is unchanged, the expected whole-iteration reduction
from auxiliary optimization is approximately 5%-10%, subject to measurement.

Recent four-H100 runs take approximately 10.3-10.6 seconds per iteration, or
roughly 340-350 iterations per hour. Auxiliary optimization alone is therefore
expected to produce approximately 310-330 iterations per hour.

The existing one-H100 atlas completed 32,256 paired continuations in about
1,300 seconds, approximately 24.8 pairs per second. At that measured rate, the
recommended 2,048-pair refresh costs approximately:

- 83 seconds on one H100;
- 21 seconds if sharded with ideal scaling across four H100s;
- about 1.0 second amortized per PPO iteration when refreshed every 20
  iterations.

For a synchronous generator sharded across the same four training H100s, budget
approximately 14%-20% fewer iterations overall after combining generation and
auxiliary optimization. That corresponds to roughly 275-305 iterations per
hour from a 340-350 iteration/hour baseline.

If a fifth H100 runs generation asynchronously, main-loop throughput should be
limited mostly by auxiliary optimization: approximately 5%-10% fewer
iterations, or 310-330 iterations per hour. The tradeoff is approximately 25%
more allocated GPU capacity for a four-training-plus-one-generator setup. At
the recommended refresh size and cadence, one sidecar H100 should comfortably
keep up.

Do not run synchronous generation on only one training H100 while the other
three wait. That can approach a 40%-50% throughput reduction. Either shard the
paired continuations across all training devices or use an asynchronous
sidecar.

These are planning estimates, not acceptance measurements. The preflight must
record separate rollout, PPO update, auxiliary update, and counterfactual
generation times. Freeze the full A/B configuration only after verifying that
the observed reduction is within the chosen compute budget.

## Implementation order

1. Refactor the paired-continuation evaluator into reusable training code.
2. Define the versioned buffer schema and target-scale tests.
3. Add version-safe auxiliary-head attachment and checkpoint migration.
4. Implement the three supervised losses with synthetic sign tests.
5. Add buffer sampling to PPO optimization without touching PPO accounting.
6. Add opportunity capture and periodic synchronous generation.
7. Add metrics, checkpoint state, and resume behavior.
8. Run single-device deterministic and throughput preflights.
9. Run a short control/treatment smoke A/B and inspect actor/critic gradients.
10. Freeze configs and launch the full matched A/B.
11. Add an asynchronous generator only if the synchronous measured overhead
    justifies the extra operational complexity.

## Key decisions to preserve during implementation

- The target is a paired causal outcome difference, not castle ownership or
  mechanical production alone.
- Negative counterfactuals are as important as positive ones.
- The scalar value head learns from distinct successor states; the auxiliary
  spatial head learns the pre-action build/control difference.
- The spatial build-difference head is required in the treatment implementation
  but remains training-only and does not alter competition action selection.
- Forced branches are supervised-only data.
- Ordinary PPO remains active throughout the treatment.
- The first A/B has no potential shaping, heuristic logit boost, or explicit
  build-action floor.
- Only policy-observable inputs may influence deployed behavior.
- Fresh held-out maps and source games determine success.

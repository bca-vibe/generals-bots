# Castle counterfactual PPO A/B: terminal report

## Executive result

The treatment was a strong success over this 400-iteration test. It learned to build castles naturally, became substantially stronger than the parent and control policies, and developed a critic that distinguishes useful from harmful builds much better than control. The critic still has a large absolute anti-build bias, even on held-out good builds, so the value-learning problem is improved rather than solved.

- Raw treatment 4,400 scored **0.889** against raw control 4,400 in 256 paired greedy games (227 wins, 28 losses, 1 draw).
- Raw treatment 4,400 scored **0.891** against the raw 4,000 parent.
- In categorical self-play, treatment built 7,817 castles in 4,096 games; **68.6%** of games contained a castle. Control built 12; **0.293%** of games contained one.
- On the held-out-good critic test, the fraction of builds valued above their controls rose from **2.0%** for control to **26.5%** for treatment. The mean value difference improved from **-0.442** to **-0.128**.
- Critic-versus-causal-effect Pearson correlation rose from **0.457** to **0.736**.

This is enough evidence to recommend the treatment recipe, including the zero-initialized residual build-kind gate, for the longer run. The main risk to monitor is overbuilding or overly confident kind gating, not failure to discover building.

## Training and fixed-reference evaluation

Both arms continued the same raw λ=.97 control checkpoint 4,000 for 400 iterations. Control used ordinary PPO. Treatment used counterfactual actor, successor-value, and delta supervision plus the distribution-preserving residual binary build-kind gate.

Final raw fixed-reference results (256 paired games):

| Policy | Score | W-L-D |
|---|---:|---:|
| Control 4,400 | 0.539 | 136-116-4 |
| Treatment 4,400 | 0.926 | 237-19-0 |

Treatment's fixed-reference score progressed from 0.611 at 4,050 to 0.926 at 4,400. Control remained near 0.5, ending at 0.539.

At iteration 4,400, treatment's rolling 50-iteration on-policy window contained 242,767 sampled builds across 115,654 completed games: 2.10 builds/game, with a castle in 71.0% of games. The counterfactual gradient norm was only about 2.65% of the PPO gradient norm, so ordinary on-policy PPO remained the dominant total update while counterfactual supervision supplied a targeted exploration/credit signal.

## Greedy nine-policy round robin

Every off-diagonal cell contains 256 paired games on 128 boards. Entries are row-policy score, with draws worth 0.5. Policies are raw and actions are masked argmax.

|row\\col|P4000|C4100|C4200|C4300|C4400|T4100|T4200|T4300|T4400|
|---|---|---|---|---|---|---|---|---|---|
|P4000|—|0.520|0.508|0.463|0.443|0.379|0.205|0.100|0.109|
|C4100|0.480|—|0.516|0.492|0.477|0.396|0.240|0.125|0.129|
|C4200|0.492|0.484|—|0.520|0.516|0.430|0.227|0.135|0.094|
|C4300|0.537|0.508|0.480|—|0.482|0.441|0.291|0.137|0.115|
|C4400|0.557|0.523|0.484|0.518|—|0.441|0.244|0.107|0.111|
|T4100|0.621|0.604|0.570|0.559|0.559|—|0.320|0.242|0.229|
|T4200|0.795|0.760|0.773|0.709|0.756|0.680|—|0.352|0.332|
|T4300|0.900|0.875|0.865|0.863|0.893|0.758|0.648|—|0.393|
|T4400|0.891|0.871|0.906|0.885|0.889|0.771|0.668|0.607|—|

Aggregate round-robin scores were 0.811 for T4,400, 0.774 for T4,300, 0.645 for T4,200, and 0.463 for T4,100. All controls and the parent were between 0.341 and 0.374. The smooth ordering among treatment milestones is strong evidence that this was sustained learning rather than one anomalous terminal checkpoint.

### Total row-policy castles by matchup

|row\\col|P4000|C4100|C4200|C4300|C4400|T4100|T4200|T4300|T4400|
|---|---|---|---|---|---|---|---|---|---|
|P4000|—|0|0|0|0|0|0|0|0|
|C4100|0|—|0|0|0|0|0|0|0|
|C4200|0|0|—|0|0|0|0|0|0|
|C4300|0|1|0|—|0|0|0|0|0|
|C4400|0|0|0|0|—|0|0|0|0|
|T4100|65|56|62|56|76|—|5|4|4|
|T4200|187|166|202|165|174|144|—|108|59|
|T4300|276|261|271|253|251|219|229|—|170|
|T4400|273|244|277|263|263|243|239|241|—|

### Percentage of games in which the row policy built

|row\\col|P4000|C4100|C4200|C4300|C4400|T4100|T4200|T4300|T4400|
|---|---|---|---|---|---|---|---|---|---|
|P4000|—|0.0%|0.0%|0.0%|0.0%|0.0%|0.0%|0.0%|0.0%|
|C4100|0.0%|—|0.0%|0.0%|0.0%|0.0%|0.0%|0.0%|0.0%|
|C4200|0.0%|0.0%|—|0.0%|0.0%|0.0%|0.0%|0.0%|0.0%|
|C4300|0.0%|0.4%|0.0%|—|0.0%|0.0%|0.0%|0.0%|0.0%|
|C4400|0.0%|0.0%|0.0%|0.0%|—|0.0%|0.0%|0.0%|0.0%|
|T4100|25.4%|21.1%|23.4%|20.3%|29.3%|—|2.0%|1.6%|1.6%|
|T4200|63.3%|56.2%|66.0%|52.7%|60.5%|49.6%|—|36.7%|21.5%|
|T4300|81.2%|78.9%|82.8%|75.8%|82.0%|73.0%|68.4%|—|48.4%|
|T4400|86.7%|78.9%|85.2%|79.7%|82.8%|77.7%|70.3%|66.8%|—|

All recorded build actions in the terminal categorical audit matched confirmed new-castle constructions exactly.

## Greedy versus stochastic building

| Audit | Control 4,400 | Treatment 4,400 |
|---|---:|---:|
| Greedy direct matchup: castles | 0 / 256 games | 263 / 256 games |
| Greedy direct matchup: games where policy built | 0.0% | 82.8% |
| Categorical self-play: castles | 12 / 4,096 games | 7,817 / 4,096 games |
| Categorical self-play: games with either side building | 0.293% | 68.6% |
| Categorical self-play: player-games with a build | 0.146% | 59.6% |

These are not identical settings: the greedy figure is treatment versus control, while the stochastic figure is same-policy self-play. Still, the qualitative result is robust. The treatment's building is no longer dependent on stochastic tail sampling; it is often the greedy decision.

Treatment self-play was also shorter: mean 482.9 and median 450 turns, versus control mean 592.5 and median 602. The truncation rate fell from 1.34% to 0.39%.

On legal-build steps, aggregate build probability averaged 0.667% for treatment versus 0.000409% for control. Treatment concentrated building early: 6,991 of 7,817 sampled builds occurred before turn 400. Although its mean best-build margin across every legal opportunity remained -13.1, the median player-game's maximum build margin was +1.08 and its median maximum build probability was 71.4%. This is a selective policy, not a uniform build boost.

## Critic atlas

Both atlases used the same 2,016 source states from 786 source games, with 16 paired build/control continuations per state and common random numbers within each pair. Values are on the critic's [-1, 1] scale.

| Metric | Control 4,400 | Treatment 4,400 |
|---|---:|---:|
| Mean V(build successor) - V(control successor), all states | -0.406 | -0.347 |
| Fraction above zero, all states | 4.48% | 14.01% |
| Mean delta, causally positive full-sample states | -0.421 | -0.103 |
| Fraction above zero, causally positive full-sample states | 2.28% | 26.46% |
| Mean delta, causally positive in both halves | -0.453 | -0.117 |
| Fraction above zero, causally positive in both halves | 1.34% | 29.45% |
| Held-out 8/8 good-build mean delta | -0.442 | -0.128 |
| Held-out 8/8 fraction above zero | 1.96% | 26.51% |
| Good-minus-bad critic separation | +0.224 | +0.499 |
| Good-minus-bad clustered 95% CI | [0.184, 0.265] | [0.461, 0.538] |
| Critic delta vs causal delta, Pearson | 0.457 | 0.736 |
| Critic delta vs causal delta, Spearman | 0.355 | 0.632 |

The treatment critic's ranking signal is substantially better. In its top value decile, builds were causally favorable on average (+0.0249), whereas the control top decile was still slightly unfavorable (-0.0008). Treatment's worst decile was strongly harmful (-0.666), showing useful discrimination at both ends.

The absolute calibration remains pessimistic. Even among the strongest held-out good-build definition, treatment assigns a negative mean immediate value shift and calls only about one quarter positive. Thus the critic has learned much more about *which* builds are good but still treats the up-front army cost too harshly relative to delayed production and defensive value.

### Common-state/action cross-arm check

The two target policies chose the same highest-logit build action on 96.6% of common source states and the same build/control action pair on 69.2%. On the 1,368 valid states where both intervention actions matched exactly, treatment's build-minus-control critic difference was 0.064 higher than control's, with a source-game-clustered 95% interval of [0.046, 0.081]. This is direct evidence of a positive critic shift on identical interventions.

The primary atlas remains a matched-policy comparison: each arm selected its own best build/control action and continued under its own policy. It is therefore not identical to evaluating all critics on a single frozen successor bank. The high build-action agreement and exact-action subset make the critic-evolution conclusion robust, but this methodological distinction should be retained.

## Recommendation for the larger run

Use the treatment design, including the residual build-kind gate. Counterfactual supervision solved the exploration bottleneck quickly enough that natural PPO samples became abundant; ordinary PPO then had ample opportunity to adapt tactics around castles. The monotonic treatment milestone ordering and dominance over all controls argue against this being merely forced overbuilding.

During the longer run, monitor:

- raw greedy and categorical castle rates separately;
- positive-versus-negative counterfactual build probability and residual-gate output;
- the held-out critic gap and fraction positive;
- strength against a broad learned-policy league, especially non-build specialists;
- signs of overbuilding, including early-build frequency, army paid per build, and strength conditional on number of castles.

I would retain the existing coefficient schedule initially. If build frequency keeps rising while strength plateaus or falls, anneal the counterfactual actor coefficient or gate learning rate before changing reward shaping. The critic evidence supports continuing value/delta supervision: policy exploration is now solved, but delayed castle value remains underlearned.

## Reproducibility and artifacts

- Control terminal SHA-256: `4b04cb8cade5dbf834b787a1c3e43b8076a06216a2c8089a668fb329bff191f9`
- Treatment terminal SHA-256: `a632333d485ec4ada92b22f0aa2c71b74f1e838eaec3bc1fc51946ff5e0559c4`
- Common source-state SHA-256: `bea81031d48b04446d2634ab7c1fd05e3cc586ef18aa0f8da52f6f9c86dda75e`
- Full compact raw records, per-iteration metrics, matchup JSON, critic NPZ files, logs, configs, and hashes are under `analysis/iteration_004400/` beside this report.
- W&B control run: <https://wandb.ai/bcarnold-independent/generals-bots/runs/castle-counterfactual-ppo-control-from-004000-20260805>
- W&B treatment run: <https://wandb.ai/bcarnold-independent/generals-bots/runs/castle-counterfactual-ppo-treatment-from-004000-20260805>
- W&B round robin: <https://wandb.ai/bcarnold-independent/generals-bots/runs/castle-counterfactual-ppo-control-from-004000-20260805-checkpoint-round-robin>
- W&B terminal audit: <https://wandb.ai/bcarnold-independent/generals-bots/runs/castle-counterfactual-ppo-terminal-audit-20260805>
- Hugging Face root: <https://huggingface.co/bca-vibe/generals-bot/tree/main/runs/castle_counterfactual_ppo_ab_from_004000_20260805>

The 8×H100 node was explicitly stopped after both checkpoints and the analysis bundle were exported, W&B synchronization completed, and the local 14 MB bundle passed file-by-file SHA-256 verification. The incremental experiment cost is approximately $98 at the stated $32/hour rate; the node's month-to-date total of $237.79 includes earlier work on the same node.

# Castle self-play audit — 2026-08-04

## Conclusion

Continuing the current training recipe is unlikely to produce a reliable
castle-building policy without an intervention. The latest raw policy has not
assigned exactly zero probability to building, but its exploration rate is too
small and its rare builds show no evidence of strategic selection. The EMA
policy used by league evaluation has caught up to the same collapsed behavior.

This is not explained by weak opponents making castles unaffordable. At the
latest checkpoint, a legal build existed in 3,995/4,096 raw stochastic
self-play games and 4,004/4,096 EMA games. Nevertheless, a castle was built on
either side in only 5 raw games (0.122%) and 6 EMA games (0.146%).

## Method

- One independent givemeanode H100.
- Exact final-stage competition map generator and rules.
- Exact observation memory, legal action mask, network logits, and categorical
  action sampling used by training rollouts.
- Identical seeds across checkpoints and raw/EMA policies.
- Every selected build was checked against a newly appearing castle in the
  environment's post-step, pre-reset state.
- Latest durable checkpoint during the audit: iteration 2,500,
  SHA-256 `71a8b0b95c4a8899c8adb0e1a691e96a3845f4caf82e842a8e170011cc53bfad`.
- The audit node `generals-castle-audit` was stopped after results were exported
  and locally hash-verified.

## Stochastic self-play results

| Iteration | Policy | Games | Games with any build | Game rate | Mean aggregate build probability on legal steps | Mean best-build logit margin | Median per-player-game peak build probability |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 1,313 | raw | 2,048 | 0 | 0.000% | 1.04e-6 | -19.00 | 3.72e-6 |
| 1,313 | EMA | 2,048 | 307 | 14.990% | 2.99e-4 | -7.65 | 4.02e-4 |
| 1,500 | raw | 2,048 | 0 | 0.000% | 5.35e-6 | -17.61 | 5.97e-6 |
| 1,500 | EMA | 2,048 | 67 | 3.271% | 9.15e-5 | -9.78 | 1.42e-4 |
| 2,000 | raw | 4,096 | 7 | 0.171% | 2.15e-6 | -17.53 | 7.43e-6 |
| 2,000 | EMA | 4,096 | 4 | 0.098% | 5.04e-6 | -13.42 | 2.37e-5 |
| 2,500 | raw | 4,096 | 5 | 0.122% | 2.81e-6 | -17.52 | 1.01e-5 |
| 2,500 | EMA | 4,096 | 6 | 0.146% | 2.37e-6 | -15.01 | 1.28e-5 |

The raw policy was already near the exploration floor at iteration 1,313. The
apparently healthier 1,313 EMA was lag: its game build rate fell from 15.0% to
3.27%, then to about 0.1% as EMA incorporated the collapsed raw weights. Raw
build probability fluctuated at a few parts per million rather than showing a
consistent recovery trend.

At iteration 2,500, raw self-play encountered 1,830,396 legal-opportunity
player-steps. The average combined probability of all legal build actions was
2.81e-6. The expected count from the probability mass was 5.14 and the sampled
count was 5, so the observed rarity is fully explained by the logits rather
than a sampler or environment bug.

## What the rare behavior looks like

At iteration 2,500:

- Raw first builds had median turn 790; EMA median turn 496.
- Raw builders went 2 wins / 3 losses. EMA builders went 1 win / 5 losses.
- Median raw army and land margins at the build were -8 armies and +1 tile.
- Median EMA margins were +1 army and -3.5 tiles.
- There were no draws among these latest rare-build games.
- Most legal-opportunity steps placed the best build hundreds of action ranks
  below the best non-build action. Raw's average best-build rank was 395 and
  its mean logit margin was -17.52.

There is a tiny raw tail where a build becomes competitive. In 4,096 greedy
self-play games, raw built exactly once and EMA never built. The single raw
greedy builder acted on turn 483 with build probability 0.310, while behind by
61 armies and 8 tiles, and lost nine turns later. This looks like an isolated
pathological state, not an emerging strategic rule.

The fixed league agrees: EMA greedy selected zero builds at iterations 1,600,
2,000, and 2,400 despite 455,994, 318,786, and 271,980 legal-opportunity steps.

## Why the current recipe is unlikely to recover unaided

The global rollout contains 2,097,152 player-steps per iteration. Applying the
latest raw opportunity frequency and build probability gives only about 2.3
build samples per training iteration. That is technically nonzero, but it is a
poor discovery channel for a long-horizon action:

- reward is terminal win/loss only;
- GAE lambda is 0.90, so direct credit over the castle's long payback horizon
  decays rapidly;
- PPO keeps only the top 25% by absolute advantage, so a rare bad build can be
  reinforced strongly in the negative direction while useful builds remain
  hard to distinguish;
- self-play has settled into a nearly symmetric no-castle convention, so the
  policy rarely sees either the upside of owning castles or the need to answer
  an opponent's castle economy;
- the entropy coefficient is decaying, so there is no reason to expect the
  exploration floor to rise spontaneously.

More wall-clock training could produce occasional builds, but the evidence does
not support expecting it to discover and stabilize the strategically correct
cases.

## Recommended next experiment

Branch from the same checkpoint and compare the current recipe against two
targeted changes before spending another long run:

1. Add action-kind exploration for `{move, build, pass}` when build is legal,
   with correct behavior-policy log probabilities for PPO. A small conditional
   build floor (start around 0.1-0.5%, then tune) or an action-kind entropy bonus
   would increase useful build samples by hundreds of times without requiring
   uniform exploration over all 3,970 flat actions.
2. Put build-capable play into the training distribution, not only evaluation:
   mix a modest fraction of games against a castle-economist/frozen
   build-capable policy, or use a short castle curriculum and then anneal back
   to exact competition rules.
3. Improve credit assignment for the branch: test a longer GAE lambda and a
   small, annealed potential-based signal tied to castle survival/payback or a
   counterfactual build-value auxiliary target. Avoid a raw per-castle bonus,
   which is likely to teach wasteful castle spam.

Run short A/B branches first and monitor raw plus EMA separately. Required
training metrics should include legal-step build probability mass, selected
builds, game build rate, build timing, build advantage, and builder outcomes.
The control branch is important: the goal is not merely more castles, but
builds that improve paired win rate without degrading ordinary tactics.

## Artifacts

- `selfplay_castles_iter1313_2048x2.json`
- `selfplay_castles_iter1500_2048x2.json`
- `selfplay_castles_iter2000_4096x2.json`
- `selfplay_castles_iter2500_4096x2.json`
- `greedy_selfplay_castles_iter2500_4096x2.json`
- `smoke_iter2000.json`
- `results.tar` — exported result bundle, SHA-256
  `90286c4b1a66c2e4cced86eca4ad665b163b840b54e436c4f37b84f52bfd9c01`

The reusable audit entry point is `tools/evaluate_selfplay_castles.py`.

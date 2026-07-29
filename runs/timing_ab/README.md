# Timing A/B: master vs throughput-optimizations

1×H100 (`timing-ab` node, deleted after), corrected generator on both arms,
competition_l7 hyperparameters, 45 iterations, seed 44, evals/checkpoints off,
shared JAX compilation cache. First 5 iterations dropped from timing stats.

| | median s/iter | median samples/s | host gap (iter − rollout − update) |
|---|---|---|---|
| A `master` (848dd46) | 8.820 | 59,547 | 0.121 s |
| B `throughput-optimizations` (4139e18) | 8.692 | 60,323 | (barriers off by default) |

- **Speedup on 1 GPU: +1.5%** — which is ~100% of the 1-GPU host gap (0.121 s
  measured on master; the branch recovered 0.128 s).
- **Training trajectories are iteration-for-iteration identical** across the
  two arms (same losses, episodes, W/L/D at every iteration) — the refactor
  changes performance only, not semantics.
- **Startup: 86 s → 32 s** wall-to-first-iteration for arm B via the (now
  default-on) persistent compilation cache reusing unchanged programs.

## Why 1.5% is not the 8× number

The pre-train measured a 1.71 s/iter host gap on 8×H100 — 14× the 1-GPU gap.
Eager dispatch on pmap-sharded arrays fans out per shard, so the overhead the
branch removes scales with device count; a single GPU simply doesn't have much
of it. The 1-GPU A/B therefore validates *correctness* decisively and shows
the branch removes essentially the entire host gap where measurable, but the
real-world speedup claim needs an 8× measurement: run ~30 iterations of master
vs branch at the start of the first branch-phase 8× session (~$3). If the same
"removes ~the whole gap" behavior holds there, expect ~1.6 s/10.7 s ≈ 15%; if
not, the floor is whatever fraction survives.

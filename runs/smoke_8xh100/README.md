# Run: smoke_8xh100 (pre-train, leg 1)

> **Generator provenance note (2026-07-29):** this entire run used the
> pre-fix map generator, which did not match the latest generator on the
> competition repo. The mismatch was fixed after the run (engine + config
> changes, see repo history after commit a1cf598). The run remains valid as a
> base checkpoint — training was still in the early curriculum — but all
> branch runs will use the corrected generator, so metrics/evals are not
> directly comparable across that boundary.

Base-model pre-train on a givemeanode 8×H100 node (`generals-smoke`), started as a
smoke test and promoted to the official pre-train. Target ~36 GPU-hours total:
2 h leg 1 + 2.5 h leg 2 (resumed from `latest.eqx`), then branch runs.

## Leg 1 — 2026-07-29 00:10:53–02:10:53 UTC

- Config: `smoke_8xh100.toml` — identical to `competition_l7.toml` except
  `run_name`, `checkpoint_every = 20`, `eval_every = 25`.
- Completed 557 iterations, ~390k samples/s sustained, curriculum stage 0 → 1.
- Ended by the planned 2 h SIGINT timeout; resume point `checkpoint_000540.eqx`.
- Eval vs random (512 paired games / 25 iters): 0.066 → 0.64 (stage 0),
  re-based to 0.50 at the stage-1 advance, climbing again to 0.547 by iter 550.

## Leg 2 — 2026-07-29 04:05:44–06:36:21 UTC (2.5 h, resumed from iter 540)

- Resumed `latest.eqx` (iter 540, curriculum stage 2) after leg 1's planned SIGINT;
  ran with `PYTHONUNBUFFERED=1` and `JAX_COMPILATION_CACHE_DIR=~/.jax_cache`
  (cache now populated on the node for fast future startups).
- Finished at **iteration 1260, curriculum stage 4 (final)**; 7.30M self-play
  episodes total across both legs; ~395k samples/s sustained.
- Eval vs random (512 paired games / 25 iters): stage-3/4 draw-conversion took
  the score from 0.50 to **0.874** (383W/0L/129D at iter 1250) — zero losses in
  every eval from iter 1100 on.
- A 12-iteration cProfile pass ran between legs (`profile_report_leg1.txt`):
  ~7 min of startup is XLA compilation (fixed by the persistent cache), the
  ~1.7 s/iter GPU-idle gap is eager host-side glue between pmapped phases,
  pool regeneration is heavyweight. Optimizations deferred to branch runs.

## Checkpoint inventory

| Where | What |
|---|---|
| Local `checkpoints/smoke_8xh100/` (gitignored) | `checkpoint_000540.eqx`, `checkpoint_000880.eqx`, `checkpoint_001260.eqx`, `latest.eqx` (= 1260), full `metrics.jsonl` — all sha256-verified against the node |
| givemeanode snapshot `snap-a8fmv` | Point-in-time copy of the node's whole disk at run end: all 64 checkpoints, venv, JAX compilation cache. Also the branch-run template (`create_node --from-snapshot`). |
| Node `generals-smoke` (stopped, disk free) | Same as snapshot; wakes on any run_command |
| Hugging Face `bca-vibe/generals-bot` | Pending: Blake pushes local checkpoints via `hf upload`; branch nodes pull via the `hf-generals-bot` read connection |

## Files

- `metrics_full.jsonl` — the complete run JSONL (both legs; iterations 541-557
  appear twice — leg-1 rows are discarded history after the rollback to the
  iter-540 checkpoint; dedupe keeping the LAST row per iteration, as
  `build_dashboard.py` does). `metrics_leg1.jsonl` and `metrics_leg1_merged.jsonl`
  kept for provenance.
- `train_stdout_leg1.log` / `train_stdout_leg2.log` — full stdout of each leg.
- `smoke_8xh100.toml` — the run config (copy of what the node ran).
- `dashboard.html` + `build_dashboard.py` + `merge_logs.py` — the live-metrics
  dashboard (published as a private claude.ai artifact during the run) and its
  generator: `python3 build_dashboard.py <metrics.jsonl> <out.html> [status]`.

Checkpoints (`*.eqx`, ~250 MB each) exceed GitHub's file limit and stay
gitignored; they are exported from the node to `checkpoints/smoke_8xh100/`
locally.

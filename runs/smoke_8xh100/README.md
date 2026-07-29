# Run: smoke_8xh100 (pre-train, leg 1)

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

## Files

- `metrics_leg1.jsonl` — the trainer's own full JSONL for leg 1 (authoritative,
  exported from the node; supersedes `metrics_leg1_merged.jsonl`, kept for
  provenance).
- `train_stdout_leg1.log` — full stdout via the provider's log service.
- `smoke_8xh100.toml` — the run config (copy of what the node ran).
- `dashboard.html` + `build_dashboard.py` + `merge_logs.py` — the live-metrics
  dashboard (published as a private claude.ai artifact during the run) and its
  generator: `python3 build_dashboard.py <metrics.jsonl> <out.html> [status]`.

Checkpoints (`*.eqx`, ~250 MB each) exceed GitHub's file limit and stay
gitignored; they are exported from the node to `checkpoints/smoke_8xh100/`
locally.

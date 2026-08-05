# Run index

Committed records (metrics, logs, configs, reports) for training and benchmark
runs. Model checkpoints never live in git — they are gitignored under
`checkpoints/`, mirrored to Hugging Face (`bca-vibe/generals-bot`) and to a
givemeanode volume snapshot.

| Run | What it is |
|---|---|
| [`smoke_8xh100/`](smoke_8xh100/) | **The base pre-train** (the name predates its promotion from smoke test). 8×H100, ~36 GPU-hours in two legs, 1,260 iterations, curriculum stage 4, eval 0.874 vs random. Used the **pre-fix map generator** — see its README's provenance note. Branch runs build on `checkpoint_001260` / HF `smoke_8xh100/latest.eqx`. |
| [`timing_ab/`](timing_ab/) | 1×H100 A/B of `throughput-optimizations` vs master (both on the corrected generator): +1.5% on 1 GPU = the entire 1-GPU host gap; training trajectories bit-identical; 8× confirmation pending at branch-phase start. |
| [`castle_selfplay_audit_20260804/`](castle_selfplay_audit_20260804/) | Raw/EMA stochastic self-play audit through iteration 2,500. Establishes the original castle-policy collapse and its extremely low action probability. |
| [`castle_counterfactual_atlas_iter3000/`](castle_counterfactual_atlas_iter3000/) | Paired forced-build atlas at checkpoint 3,000. Shows that indiscriminate builds are harmful but a reproducibly valuable subset is identifiable from policy-observable features. |
| [`castle_value_probe_iter3000/`](castle_value_probe_iter3000/) | Successor-state critic probe at checkpoint 3,000. Shows that the critic predicts a large value decrease even for causally beneficial builds. |
| [`castle_phi_boost_ab_lambda097_from_003003_20260804/`](castle_phi_boost_ab_lambda097_from_003003_20260804/) | Completed 1,000-iteration, 4×H100-per-arm A/B of λ=0.97 control versus actor potential shaping plus tactical build-logit boost. Includes the final raw/EMA round robin and experiment conclusion. |
| [`castle_raw_checkpoint_followup_20260804/`](castle_raw_checkpoint_followup_20260804/) | Post-A/B stochastic league and common-state critic analysis. Explains why the treatment failed to retain castle behavior. |

## Conventions

- Canonical training recipes live in `generals/training/configs/`; each run
  directory keeps a copy of the exact config it ran, so recipes can evolve
  without rewriting history.
- Dashboard/reporting tooling lives in `tools/dashboard/`, not in run
  directories; runs keep only their data and rendered snapshots.

# Castle counterfactual anneal continuation

This run continues the raw treatment checkpoint at global iteration 4,400 to
iteration 13,000 on one 8×H100 node. It preserves the prior global PPO and
counterfactual batch geometry while splitting each global batch across eight
devices.

## Supervision schedule

- Actor counterfactual preference coefficient: linear from 1.0 after 4,400 to
  0.0 at 4,500.
- Successor-value and build-difference coefficients: 1.0 through 4,500, then
  linear to 0.0 at 4,700.
- Iterations after 4,700 use the ordinary PPO updater and do not sample the
  counterfactual replay buffer or generate counterfactual continuation
  rollouts.
- Ordinary PPO actor/value losses, λ=.97, the residual build-kind head, and
  the learned actor/critic parameters continue throughout.

## Checkpoints and evaluation

- Recovery checkpoint: every 100 global iterations.
- Numbered checkpoint and paired learned-policy evaluation: every multiple of
  500, plus iteration 4,600.
- The initial raw and EMA leagues each contain the corresponding policy from
  the previous treatment checkpoint at 4,400.
- At each multiple of 1,000, the current checkpoint is evaluated first and is
  then admitted to the raw and EMA leagues separately.
- Each matchup uses 128 fixed final-curriculum maps with both seat assignments
  (256 games). Raw never plays an EMA league member and EMA never plays a raw
  league member.
- A graceful early stop or natural completion writes a terminal full training
  checkpoint and runs a 256-map paired raw-versus-EMA head-to-head.

## Durable records

Training, castle, annealing, checkpoint, and league metrics are written to
`metrics.jsonl` and streamed to W&B. Every learned-league evaluation has a
standalone `learned_league_XXXXXX.json`; admitted members and hashes are kept
in `learned_league_manifest.json`. Checkpoint publication bundles contain the
full training state, raw and EMA competition exports, manifests, hashes, and
counterfactual replay shards only while those shards remain relevant.

The authoritative configuration is
`generals/training/configs/castle_counterfactual_anneal_long_from_4400.toml`.

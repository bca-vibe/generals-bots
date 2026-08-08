# 26k EMA inference-temperature sweep

## Result

**Temperature 0.05 is the best stochastic setting by point estimate, but it did
not establish a statistically significant advantage over greedy play.** For a
stochastic competition candidate, use `T=0.05`; retain greedy as the conservative
control because the measured difference is small.

Across the complete 25,600-game dataset, an inverse-variance weighted paired-map
model estimates the following score against greedy:

| Rank | Inference | Modeled score vs greedy | Approx. 95% CI |
| ---: | --- | ---: | ---: |
| 1 | T=0.05 | 50.76% | 49.33%-52.20% |
| 2 | T=0.15 | 50.59% | 49.15%-52.04% |
| 3 | Greedy | 50.00% | reference |
| 4 | T=0.25 | 49.24% | 48.09%-50.39% |
| 5 | T=0.20 | 49.14% | 47.69%-50.58% |
| 6 | T=0.10 | 49.08% | 47.92%-50.24% |
| 7 | T=0.50 | 48.88% | 47.32%-50.43% |
| 8 | T=1.00 | 45.97% | 44.39%-47.56% |

The direct fine-stage matchup had T=0.05 scoring 51.22% against greedy over
1,024 games (paired-map 95% CI 48.62%-53.82%). T=0.15 directly beat T=0.05
51.95%-48.05%, so the data supports a shallow low-temperature region rather
than a sharply identified optimum. Temperature 1.0 is clearly too hot.

## Stages

The broad round robin used greedy, 0.10, 0.25, 0.50, and 1.00. Each agent
played 4,096 games, and greedy ranked first at 52.28% macro score.

| Rank | Inference | Macro score | W-L-D |
| ---: | --- | ---: | ---: |
| 1 | Greedy | 52.28% | 2105-1918-73 |
| 2 | T=0.10 | 50.44% | 2020-1984-92 |
| 3 | T=0.25 | 50.32% | 2024-1998-74 |
| 4 | T=0.50 | 50.29% | 2021-1997-78 |
| 5 | T=1.00 | 46.67% | 1877-2150-69 |

The fine round robin used greedy and temperatures 0.05 through 0.25 in 0.05
increments. Each agent played 5,120 games.

| Rank | Inference | Macro score | W-L-D |
| ---: | --- | ---: | ---: |
| 1 | T=0.05 | 51.13% | 2567-2451-102 |
| 2 | T=0.15 | 50.98% | 2568-2468-84 |
| 3 | Greedy | 49.77% | 2506-2530-84 |
| 4 | T=0.25 | 49.66% | 2491-2526-103 |
| 5 | T=0.10 | 49.28% | 2471-2545-104 |
| 6 | T=0.20 | 49.19% | 2468-2551-101 |

## Protocol and provenance

- Policy: iteration 26,000 EMA from
  `bca-vibe/generals-bot@main:runs/castle_ppo_gmn_8xh100_original_from_021001_to_027000_20260807/checkpoints/iteration_026000/training_checkpoint.eqx`
- Checkpoint SHA-256:
  `fa98e35d79b0d3637dbc90408da69d1b9b3c2113d1210f392b53e6f4fb37c81e`
- Exported EMA weights SHA-256:
  `9fb390390e9317d69c6448fd23317e7ad5e70aa37545823b7043405eaa150bac`
- 1,024 games per matchup: 512 locked maps with both seat assignments
- 10 broad matchups plus 15 fine matchups = 25,600 total games
- Broad seeds: map `202608081`, action `202608082`
- Fine seeds: map `202608083`, action `202608084`
- Four shards of 128 maps (256 games) per matchup
- Hardware: secure RunPod A100-SXM4 80 GB; JAX 0.11.0
- Measured evaluation time: 1,345.5 seconds (22m 25.5s), excluding setup
- The RunPod was deleted after all outputs were copied locally.

Sampling uses categorical draws from `softmax(logits / temperature)`. Greedy is
represented as temperature zero in the sweep inputs but uses `argmax`, not a
near-zero softmax. All competitors use the exact same exported EMA weights.

The combined model fits one strength per temperature to the paired-map score
logits using inverse-variance weighted least squares, anchored at greedy. Its
intervals are approximate and do not correct for selecting the winner after the
sweep. Raw direct-match results remain the primary evidence.

## Artifact index

- `broad.json`, `fine.json`: complete machine-readable round-robin outputs,
  including every matchup, confidence interval, timing, aggregate behavior
  counters, matrices, participant hashes, and rankings
- `matchups.csv`: flattened per-matchup data for analysis and plotting
- `broad_ranking.csv`, `fine_ranking.csv`: stage rankings
- `combined_model.json`, `combined_ranking.csv`: combined fit and diagnostics
- `repeated_direct_pairs.csv`: independently repeated pairings across both stages
- `combined_ranking.svg`: compact result visualization
- `export_metadata.json`: checkpoint/export provenance
- `temperature_sweep.py`: GPU evaluation driver
- `analyze.py`: deterministic aggregation and chart generation
- `verify_artifacts.py`, `SHA256SUMS`: integrity and completeness checks

`temperature_sweep.py` imports the shared evaluation helpers from
`runs/inference_temperature_sweep_20260807/coarse_14k/round_robin.py`; place that
directory on `PYTHONPATH` when reproducing a run. Regenerate the derived files
with:

```bash
python runs/ema26k_temperature_sweep_20260808/analyze.py
python runs/ema26k_temperature_sweep_20260808/verify_artifacts.py
```

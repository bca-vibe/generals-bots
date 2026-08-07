# Incomplete three-checkpoint screen

This stage was intentionally stopped after 33 of 105 planned matchups because
the checkpoint ordering was already consistent and the remaining GPU time had
higher decision value in a checkpoint-14,000 temperature search.

The partial artifact contains 33,792 games. Every completed 12k-vs-10k cell
favored 12k (14/14, mean score 59.19%), and every completed 14k-vs-10k cell
favored 14k (10/10, mean score 62.80%). No 14k-vs-12k cell had been reached in
this schedule. Do not treat the partial macro ranking as balanced: policies
had unequal opponent coverage when the run stopped.

- `round_robin.json` contains every completed matchup, all behavior counters,
  seeds, hashes, and the incomplete 15-policy score matrix.
- `matchups.csv` is the flattened one-row-per-matchup representation.
- `round_robin.py` is the exact evaluator source.
- `run.log` contains the complete progress log through the stop point.

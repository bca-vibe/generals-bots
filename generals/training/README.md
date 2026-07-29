# Competition L7 baseline training

This package implements the shared pre-training baseline for the Generals Bot
competition. It is intentionally close to AverageJoe's released `L_7d` model,
with only the changes required by the competition environment.

## Architecture

The actor receives a deterministic 38-channel, 21×21 observation and two
512-turn opponent-score histories. The spatial observation is divided into 49
non-overlapping 3×3 patches and projected from 342 to 448 values per token.
Two history tokens and one learned value token produce a 52×448 sequence.

The torso contains seven pre-LayerNorm transformer blocks:

- Eight attention heads with 56 values per head.
- 448-dimensional token representations.
- A 1,344-dimensional SiLU feed-forward layer.
- No dropout or recurrence.
- Learned absolute position embeddings.

The normalized spatial tokens pass through a shared `448 -> 81` projection,
which unpatchifies into eight movement planes plus one castle-build plane. A
separate `448 -> 1` projection from the value token produces one canonical pass
logit. The final masked policy contains exactly 3,970 actions. The value token
also produces 128 categorical logits over `[-1, 1]`.

The model has 15,345,362 trainable floating-point parameters.

## Observation memory

`observation.py` maintains deterministic, per-player episode state:

- Seven own-army and visible-enemy-army delta frames.
- Persistent seen terrain, structures, and previously observed enemy regions.
- Last visible enemy army and logarithmic time-since-seen.
- Public opponent army and land totals over 512 turns.
- Absolute coordinates and broadcast game statistics.

Rectangular maps carry a public board mask. Padding is encoded as a known
mountain rather than fog, matching the dimensions supplied to a submitted bot
in the protocol handshake.

## Action representation

Flat action indices are ordered as:

1. Four all-but-one movement planes (`4 * 441`).
2. Four half-army movement planes (`4 * 441`).
3. One build plane (`441`).
4. One global pass action.

Build legality and price are calculated from the observation only. This is
exact because a player's own general, castles, cells, and armies are always
visible, and enemy structures never contribute to that player's build price.

## PPO training

The training loop uses symmetric current-policy self-play. Both player seats
are included as samples. The production configuration uses:

- Sparse terminal rewards: win `+1`, loss `-1`, draw `0`.
- `gamma=1.0`, GAE lambda `0.90`.
- PPO clipping `0.20`, one epoch per rollout.
- The top 25% of samples by absolute normalized advantage.
- HL-Gauss categorical value loss with 128 bins and sigma `0.04`.
- Power-law learning-rate and entropy schedules.
- Global gradient clipping at `0.267`.
- EMA parameters with decay `0.999`.
- Data-parallel gradient averaging across every visible accelerator.

The curriculum changes only general separation. Map sizes, fog, castle
building, deathtouch, the draw horizon, and every other competition rule stay
fixed. A stage advances when the EMA policy reaches a 60% paired-map score
against the frozen uniformly random evaluator.

## Installation and launch

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train]"

generals-train \
  --config generals/training/configs/competition_l7.toml
```

The same entry point is available without installing the console script:

```bash
python -m generals.training.train \
  --config generals/training/configs/competition_l7.toml
```

Checkpoints and `metrics.jsonl` are written beneath
`checkpoints/competition_l7_baseline/`. Resume all model, optimizer, EMA, and
curriculum-stage state with:

```bash
generals-train \
  --config generals/training/configs/competition_l7.toml \
  --resume checkpoints/competition_l7_baseline/latest.eqx
```

A resumed run begins with fresh self-play environments and deterministic
observation memory; the checkpoint restores the host RNG used to generate
subsequent pools and evaluations.

The checked-in iteration count is a generous ceiling, not a prescribed compute
budget. Stop and resume based on measured accelerator-hours and evaluation
curves. Before paying for a long run, benchmark the exact configuration on the
chosen provider and verify checkpoint recovery after an intentional stop.

## Outputs to monitor

Every metrics record contains rollout and update time, samples/second, PPO
losses, entropy, approximate KL, clipping fraction, gradient norm, value
explained variance, and completed self-play outcomes. Evaluation records report
paired-map win/loss/draw counts, expected score, and paired-score dispersion.

The random opponent is only a curriculum gate and pipeline diagnostic. It is
not sufficient for selecting the final model; architecture branches should be
compared against a frozen checkpoint league on locked paired maps.

# Competition L7 transformer training

This package implements the shared pre-training baseline for the Generals Bot
competition. It is intentionally close to AverageJoe's released `L_7d` model,
with only the changes required by the competition environment.

## Architectures

New competition runs use a deterministic 37-channel, 21×21 observation and two
512-turn opponent-score histories. The spatial observation is divided into 49
non-overlapping 3×3 patches and projected from 333 to 448 values per token. Two
history tokens and one learned value token produce a 52×448 sequence. The
versioned `legacy_38` schema remains available for historical checkpoints.

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

Two matched model variants are available:

- `transformer` is the pure 7-layer transformer control, with approximately
  15.34 million trainable parameters under `competition_37`.
- `conv_transformer` adds a 96-channel convolutional residual before the first
  transformer block. Overlapping 3×3 convolutions build cell-level local
  features, downsample them to 49 corrections, and add them to the ordinary
  patch tokens before absolute position embeddings. It adds 787,744 trainable
  parameters; the rest of the architecture is unchanged.

For a fresh convolutional run, the output projection is calibrated once on 512
two-seat observations from the generated competition-state pool. Its weights
and bias are rescaled so the convolutional correction has 10% of the ordinary
patch tokens' RMS immediately before the streams are added. The ratio is not
constrained after initialization:

```text
RMS(delta_conv) / RMS(patch_tokens) = 0.10
```

The calibration batch contains 256 freshly generated maps viewed from both
player seats, using the same normalized `competition_37` inputs as training.
The corresponding configuration fields are
`conv_initial_token_rms_ratio = 0.10` and `conv_calibration_samples = 512`.
Calibration statistics—including the before/after ratios and applied projection
multiplier—are written to `conv_calibration.json`. Checkpoint resumes restore
the learned projection and do not recalibrate it. Given the same training seed,
the transformer parameters in both variants remain bit-identical before
training; only the convolutional parameters are extra.

## Observation memory

`observation.py` maintains deterministic, per-player episode state:

- Seven own-army and visible-enemy-army delta frames.
- Persistent seen terrain, structures, and previously observed enemy regions.
- Persistent plain-cell evidence used to infer newly built castles in fog.
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

### Optional Weights & Biases reporting

The local `metrics.jsonl` file remains the canonical metrics record. To mirror
training, evaluation, configuration, console, and host/GPU telemetry to W&B,
install the optional tracking dependency and set a project:

```bash
pip install -e ".[train,tracking]"
WANDB_PROJECT=generals-bots generals-train \
  --config generals/training/configs/competition_l7.toml
```

`WANDB_API_KEY` and `WANDB_ENTITY` use the standard W&B environment variables.
Set `WANDB_MODE=offline` to record locally for a later `wandb sync`, or
`WANDB_MODE=disabled` to force tracking off. A recipe can instead set
`wandb_project`, `wandb_entity`, `wandb_group`, and `wandb_tags` in its
`[training]` table; environment `WANDB_PROJECT` enables tracking when the
recipe leaves `wandb_project` unset.

Every process invocation is a separate W&B run. Resumed legs are named with
their checkpoint iteration and grouped under `wandb_group`, or under
`run_name` by default. This preserves rollback provenance when a checkpoint is
older than metrics already emitted by an earlier leg. W&B failures disable the
remote sink without interrupting training; JSONL logging continues normally.
Checkpoints are not uploaded to W&B.

Launch the matched convolutional variant with:

```bash
generals-train \
  --config generals/training/configs/competition_l7_conv.toml
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

Convolutional checkpoints are written beneath
`checkpoints/competition_l7_conv_stem/` and must be resumed with
`competition_l7_conv.toml`. Checkpoints cannot be resumed across observation
schemas or model architectures.

```bash
generals-train \
  --config generals/training/configs/competition_l7_conv.toml \
  --resume checkpoints/competition_l7_conv_stem/latest.eqx
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
Fresh convolutional runs additionally write `conv_calibration.json`; its
`ratio_after` should be approximately `0.10` before the first PPO iteration.

The random opponent is only a curriculum gate and pipeline diagnostic. It is
not sufficient for selecting the final model; architecture branches should be
compared against a frozen checkpoint league on locked paired maps.

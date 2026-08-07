# Continuation training runbook

This is the operational guide for continuing a production Generals Bot training
run on either GiveMeANode or Google Cloud. It is written for a coding agent that
has repository access but no prior context from the training conversation.

The overriding rule is: **resume from the newest verified, durable full-training
checkpoint, using the exact source, model configuration, Python/JAX runtime, and
league files that checkpoint expects.** A W&B iteration is not a checkpoint, a
competition ZIP is not resumable training state, and an unverified file on an
ephemeral GPU host is not durable.

## Current production example

As of 2026-08-06, the active run is:

- Run: `castle_counterfactual_anneal_long_from_004400_20260805`
- Authoritative config:
  [`generals/training/configs/castle_counterfactual_anneal_long_from_4400.toml`](generals/training/configs/castle_counterfactual_anneal_long_from_4400.toml)
- W&B source run:
  `bcarnold-independent/generals-bots/castle-counterfactual-anneal-long-from-004400-20260805`
- Hugging Face root:
  `bca-vibe/generals-bot@main:runs/castle_counterfactual_anneal_long_from_004400_20260805`
- Newest durable checkpoint at the time this file was written: iteration 9,000
- Full checkpoint path:
  `runs/castle_counterfactual_anneal_long_from_004400_20260805/checkpoints/iteration_009000/training_checkpoint.eqx`
- Checkpoint SHA-256:
  `b7e397cec2d226970a223749f34355541ec147b560857c7daf2933bb039580df`

The run is still active and produces a numbered checkpoint every 500 iterations,
so **do not assume 9,000 is still latest**. Discover and verify the newest
completed export before launching a continuation.

## 1. Decide what kind of continuation this is

There are two distinct cases.

### Infrastructure recovery of the same logical run

Use this when a host was lost or stopped unexpectedly and the training recipe is
unchanged.

- Resume from the newest checkpoint that was exported and hash-verified, even if
  W&B shows later iterations.
- Restore the same run directory layout and learned-league checkpoints.
- Repeating iterations after the last durable checkpoint is expected.
- Prefer a new W&B leg/run ID in the same W&B group. Reusing the old W&B run ID
  can create duplicate iteration rows when replaying lost work.
- If the old W&B ID must be reused, regard local `metrics.jsonl` as canonical,
  use last-record-wins by `iteration`, and backfill a clean derived run with
  [`tools/backfill_deduplicated_wandb.py`](tools/backfill_deduplicated_wandb.py)
  after training.

### A new continuation experiment

Use this when changing the target iteration, schedules, reward design, model
behavior, league composition, or any other training parameter.

- Copy the parent TOML to a new config file.
- Give it a new `run_name`, output directory, W&B run ID/name, and Hugging Face
  destination.
- Record the parent checkpoint iteration, sample count, source path, SHA-256,
  policy/optimizer/EMA restore choices, curriculum stage, and device geometry.
- Never silently edit the config or artifacts of a completed parent run.

## 2. Select the checkpoint safely

Select a checkpoint using durable storage, not the most optimistic progress
indicator.

1. Enumerate completed numbered checkpoints in the Hugging Face run root, or
   publication directories on a still-running source node.
2. Require all of the following:
   - `training_checkpoint.eqx`
   - `manifest.json`
   - `checkpoint_schema.json` and the counterfactual sidecar when the model has
     counterfactual heads
   - raw and EMA competition bundles when publication is expected to include
     them
   - a completed export marker or provider-confirmed completed export
3. Read the expected SHA-256 from the manifest and verify the downloaded bytes:

   ```bash
   sha256sum training_checkpoint.eqx
   ```

4. Confirm the checkpoint's saved iteration, curriculum stage, observation
   schema, architecture, and counterfactual/residual-head schema against the
   continuation config.
5. Resume from `training_checkpoint.eqx`, never `competition_raw.zip`,
   `competition_ema.zip`, or exported inference weights. The full checkpoint
   contains raw parameters, optimizer state, EMA parameters, global iteration,
   curriculum stage, and host RNG.

If a source node has a later local checkpoint than Hugging Face, it is usable
only after its file hash and metadata have been checked and it has been copied
to durable storage. W&B progress alone is not recoverable state.

## 3. Preserve the model and batch geometry

The current production checkpoint requires:

- `observation_schema = "competition_39"`
- `model_architecture = "conv_transformer"`
- depth 7, embedding dimension 448, 8 attention heads
- convolutional channels/groups 96/12
- BF16 execution
- 128-bin value head
- residual build-kind head and counterfactual auxiliary head present
- curriculum stage 4

The current eight-GPU geometry is:

- 256 environments per GPU, 2,048 globally
- PPO minibatch size 512 per GPU, 4,096 globally
- rollout length 512
- one PPO epoch

Configuration values such as `num_envs` and `minibatch_size` are **per device**.
If device count changes, preserve the global geometry unless the experiment
explicitly intends to change it:

```text
new per-device value = old global value / new device count
```

For either provider's single eight-H100 host, retain the checked-in per-device
values. Do not multiply them by eight again.

The counterfactual actor and critic schedules reached zero at iterations 4,500
and 4,700 respectively. Continuing this run after that point should report:

```text
counterfactual/actor_total_scale  = 0
counterfactual/critic_total_scale = 0
```

No counterfactual generation rollouts should run once both scales are zero.
The residual build-kind head remains part of the policy and must still load.

## 4. Required runtime

Do not install an unpinned generic `jax` package and assume GPU execution is
correct. The recovered production trajectory is validated with:

- Python 3.11.15
- JAX 0.10.2
- jaxlib 0.10.2
- `jax-cuda12-plugin` 0.10.2
- `jax-cuda12-pjrt` 0.10.2
- a CUDA 12.9 toolchain/`ptxas` for the known-good environment
- cuDNN and NCCL available to JAX
- eight visible NVIDIA H100 devices

Application dependencies come from `uv.lock` and the `train` plus `tracking`
extras. In practical terms the environment must include NumPy, pygame,
`python-socketio[client]`, Equinox, Optax, W&B, JAX, jaxlib, the CUDA plugin,
and PJRT. The host or image also needs CA certificates, `curl`, and `git` for
source/artifact transfer.

Use [`tools/verify_training_runtime.py`](tools/verify_training_runtime.py) to
fail fast:

```bash
python tools/verify_training_runtime.py \
  --python 3.11 \
  --jax 0.10.2 \
  --jaxlib 0.10.2 \
  --devices 8
```

The output must name eight `CudaDevice` entries. A successful `nvidia-smi` is
necessary but not sufficient; JAX itself must see all eight devices.

### Preferred installation: prebuilt image

Build or use [`Dockerfile.givemeanode`](Dockerfile.givemeanode). Despite its
name, it is also the preferred portable runtime for Google Cloud. It pins:

- `nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04`
- uv 0.11.15
- Python 3.11.15
- the locked project dependencies
- `jax[cuda12-local]==0.10.2`

Building the image before allocating expensive GPUs avoids spending H100 time
downloading and resolving packages. Push the image to a registry accessible by
the selected provider and record its immutable digest.

The Dockerfile's default `CMD` targets the historical A/B supervisor. Override
the container command with the continuation-supervisor command in section 11;
do not run the image with its default command for a single continuation.

The Docker build context **must contain `README.md`** because `pyproject.toml`
declares it as project metadata. Omitting it and then running an editable install
causes project installation to fail.

### Fallback installation on a fresh host

Use [`tools/bootstrap_givemeanode_training_env.sh`](tools/bootstrap_givemeanode_training_env.sh),
which is provider-agnostic despite its filename:

```bash
CUDA_MODE=local \
EXPECTED_GPU_COUNT=8 \
VENV_PATH="$PWD/.venv" \
tools/bootstrap_givemeanode_training_env.sh
```

Use `CUDA_MODE=local` only when the base image already has compatible CUDA,
cuDNN, NCCL, and `ptxas`. Otherwise use the portable bundled mode explicitly:

```bash
CUDA_MODE=bundled \
ALLOW_BUNDLED_CUDA_DOWNLOAD=1 \
EXPECTED_GPU_COUNT=8 \
VENV_PATH="$PWD/.venv" \
tools/bootstrap_givemeanode_training_env.sh
```

Bundled mode downloads roughly 3.1 GiB of CUDA wheels. During the 2026-08-06
recovery, an external package-CDN path made this take far longer than expected.
Do not discover this after an expensive node starts billing: prefer the image,
a provider-local wheel cache, or a prebuilt wheelhouse.

For a deliberately minimal source archive that omits `README.md`, do **not** run
`pip install -e .`. The bootstrap script uses:

```bash
uv sync --locked --no-install-project --extra train --extra tracking
```

Run from the repository root with `PYTHONPATH` set to that root. This keeps
minimal recovery contexts valid without weakening dependency pinning.

## 5. Source and storage layout

Use an exact source revision or a deterministic source archive. Exclude `.git`,
local virtual environments, caches, old checkpoints, and unrelated `runs/`
artifacts from transfers. Record the archive or commit SHA.

Recommended layout:

```text
WORK_ROOT/
  generals-bots/                 exact source tree
  .venv/                         pinned runtime, if not using a container
  .cache/jax/                    bounded task-local compilation cache
  runs/RUN_NAME/                 metrics, checkpoints, evals, publications
```

On Google Cloud, place `runs/RUN_NAME` and any only copy of downloaded
checkpoints on a persistent disk, not Local SSD. A3 Local SSD is useful for
cache/scratch data but must not be the sole home of a checkpoint. On
GiveMeANode, verify the mounted disk and free space before downloading; do not
attach an old training volume or large snapshot unless its contents are
actually needed.

Before launch, verify:

```bash
nvidia-smi -L
df -h
mount
```

Budget at least 40 GB for a clean runtime and working files. The current run's
minimal working set is much smaller than a large training snapshot; avoid
copying unrelated historical checkpoints.

## 6. Restore learned-league state

The learned league is reconstructed from numbered full checkpoints in the run
directory, not merely from `learned_league_manifest.json`.

For this run, restore:

1. The 4,400 anchor at the exact local path configured by
   `learned_league_anchor_path`:

   ```text
   runs/castle_counterfactual_ppo_treatment_from_004000_20260805/terminal.eqx
   ```

2. Every already-admitted 1,000-iteration checkpoint between the anchor and the
   resume point, named in the continuation run directory as:

   ```text
   checkpoint_005000.eqx
   checkpoint_006000.eqx
   checkpoint_007000.eqx
   checkpoint_008000.eqx
   checkpoint_009000.eqx
   ...
   ```

3. The resume checkpoint itself, even when it is not a league-admission
   multiple.

Verify every hash. If an admitted checkpoint is absent, the loader silently
skips that league member, changing evaluation methodology. Therefore compare
the reconstructed `learned_league_manifest.json` to the expected member list
before accepting the launch.

Only the active pair of networks is kept on GPU during a matchup, so restoring
these checkpoint files does not imply loading every league member into GPU
memory simultaneously.

## 7. Prepare a continuation config

Copy the parent config and update, at minimum:

- `run_name`
- `wandb_group`, `wandb_run_id`, `wandb_run_name`, and tags
- `parent_wandb_run_id` and URL
- `parent_final_iteration`
- `parent_final_samples`
- `resume_checkpoint_source`
- `resume_checkpoint_sha256`
- `resume_checkpoint_has_counterfactual_heads = true`
- `resume_start_stage = 4`
- previous/current device geometry provenance
- `num_iterations`
- checkpoint/evaluation cadence
- Hugging Face destination passed to the supervisor

Keep these restore flags true for an ordinary continuation:

```toml
resume_raw_weights = true
resume_optimizer_state = true
resume_ema_weights = true
```

For the current global geometry, each completed iteration contains 2,097,152
agent samples. If reconstructing provenance from a known parent, compute
`parent_final_samples` from the authoritative metrics rather than guessing.

Parse and validate the TOML on CPU before touching the GPUs. A config/checkpoint
architecture mismatch will prevent Equinox deserialization; do not work around
that error by weakening validation.

## 8. Credentials and remote logging

Credentials must exist before production starts.

### GiveMeANode

Use stored connections/secrets rather than putting tokens in commands:

- W&B secret: `wandb-prod`
- Hugging Face read/write connection: `hf-generals-bot-write`

Pass `WANDB_API_KEY` through the command's secret environment mapping. Use the
Hugging Face connection for checkpoint imports and publication exports.

### Google Cloud

Store W&B and Hugging Face tokens in Secret Manager and expose them only to the
training service process. Do not put tokens in instance metadata, shell history,
Docker build arguments, source archives, or command lines. Use a narrowly scoped
VM service account for Google Cloud Storage/Artifact Registry access.

Before launching PPO, perform a lightweight W&B initialization/finish check and
confirm the intended entity, project, group, run ID, and URL. Confirm that the
Hugging Face destination repository is writable. Do not upload checkpoints to
W&B; it is metrics/telemetry, not the checkpoint store.

Local `metrics.jsonl` remains authoritative even when W&B is healthy.

## 9. Provider-specific provisioning

### GiveMeANode

1. Request one 8×H100 node; queued time is free.
2. Prefer a prebuilt image based on `Dockerfile.givemeanode`.
3. If using a source context, upload only the small source/config archive and
   pull checkpoints through the Hugging Face connection.
4. Inspect mounts and disk use before downloads. Abort on an unexpected
   persistent volume or an unexpectedly large transfer plan.
5. Keep caches task-local and bounded.
6. Run production detached and retain the returned command ID for monitoring.
7. GiveMeANode bills running and idle nodes, so stop the node promptly only
   after all artifacts are durably exported.

### Google Cloud

Use a single `a3-highgpu-8g` Compute Engine VM for eight H100 80 GB GPUs. Follow
Google's current [A3 creation documentation](https://cloud.google.com/compute/docs/gpus/gpudirect)
and [GPU driver guidance](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)
rather than copying a stale `gcloud` command from this file.

Recommended approach:

1. Confirm regional H100 quota and actual zonal capacity before planning the
   run.
2. Use an Ubuntu 22.04/24.04 accelerated image with drivers installed, or a
   supported Container-Optimized OS plus its documented GPU-driver setup.
3. Run the pinned CUDA 12.9.1 training container by immutable digest.
4. Attach a persistent disk for run state and mount it before launch.
5. Use Local SSD only for caches or replaceable scratch data.
6. Use a systemd service or equivalent supervisor so logs and exit status
   survive SSH disconnection.
7. Enable termination handling that sends `SIGTERM` to the continuation
   supervisor and allows it to save terminal state. Spot/Flex Start can be used
   only if the persistent-disk and frequent-export recovery path has been
   tested; on-demand is safer for the final run.

Google documents A3 High as the H100 family and currently recommends CUDA
12.2.2 or later with a compatible NVIDIA driver. This project deliberately pins
a newer CUDA 12.9 userspace; the runtime verifier and an actual JAX device
operation are the acceptance test.

## 10. Minimal paid-GPU preflight

Do not run JAX tests on a local machine without the required capacity. Perform
GPU checks only on the provisioned host.

For an unchanged, previously validated source/config/runtime combination, avoid
a separate compiled one-iteration smoke that duplicates compilation cost. The
minimum preflight is:

1. Eight H100s visible in `nvidia-smi`.
2. Runtime verifier reports the exact Python/JAX/plugin/PJRT versions and eight
   JAX CUDA devices.
3. Config parses and validates.
4. Checkpoint SHA, architecture, schema, saved iteration, stage, and auxiliary
   heads match.
5. Anchor and admitted league checkpoints exist with correct hashes.
6. W&B smoke succeeds and yields the expected project/run metadata.
7. Hugging Face read and write paths are accessible.
8. At least 40 GB disk is available and the run directory is on durable storage.

Then launch production and treat its first completed iteration as the compiled
training smoke. Confirm finite losses, eight-device execution, nonzero
throughput, and the expected next global iteration before declaring the run
live.

Use [`generals.training.continuation_preflight`](generals/training/continuation_preflight.py)
only when the checkpoint format, model code, device topology, or provider
runtime has materially changed and the extra compilation/run cost is justified.

## 11. Launch through the CPU-only supervisor

Always launch via
[`generals.training.continuation_supervisor`](generals/training/continuation_supervisor.py).
It runs PPO as a child, forwards graceful-stop signals, and prepares publication
bundles without blocking the PPO update path.

The parent supervisor must not initialize JAX on a GPU. Use:

```bash
CUDA_VISIBLE_DEVICES="" \
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TRAIN_EXPECTED_JAX_DEVICE_COUNT=8 \
PYTHONPATH="$PWD" \
JAX_COMPILATION_CACHE_DIR="$PWD/.cache/jax" \
PYTHONUNBUFFERED=1 \
python -u -m generals.training.continuation_supervisor \
  --config generals/training/configs/CONTINUATION.toml \
  --resume /durable/path/checkpoint_XXXXXX.eqx \
  --duration-hours 12 \
  --hf-root OWNER/REPO@main:runs/NEW_RUN
```

The empty parent `CUDA_VISIBLE_DEVICES` is intentional. The supervisor removes
the `TRAIN_` prefixes for the child, which then receives all eight GPUs and
`EXPECTED_JAX_DEVICE_COUNT=8`. Running the supervisor with GPUs visible can make
the parent reserve memory on GPU 0 before training starts.

Launch the command detached through GiveMeANode, or as a managed systemd service
on Google Cloud. Record the command/service ID and log path.

Do not declare success merely because the process exists. Confirm:

- the child prints eight `CudaDevice` entries
- the checkpoint resumes at the expected iteration and stage
- the first new iteration completes
- W&B receives the first new metrics
- counterfactual total scales have the intended values
- samples/second and iteration time are plausible for the known workload

## 12. Checkpoint publication is a separate responsibility

The continuation supervisor prepares local bundles under:

```text
runs/RUN_NAME/publish_ready/iteration_XXXXXX/
```

It does not, by itself, guarantee that Hugging Face received them. An external
monitor must:

1. Find every publication directory without `.hf_export_complete`.
2. Upload the entire directory to the corresponding Hugging Face iteration
   path.
3. Poll the provider operation to confirmed completion.
4. Verify bytes/hash/manifest as supported by the provider.
5. Only then create `.hf_export_complete`.

Transient Hugging Face LFS negotiation or verification failures can return HTTP
504. Keep the local checkpoint, wait, and retry with backoff. Never create the
completion marker after a failed or ambiguous upload.

On Google Cloud, the equivalent monitor can use `huggingface-cli`/`hf upload`
with a secret-injected token, and can additionally mirror full checkpoints to
Cloud Storage. Do not delete the VM or persistent disk until the remote copy is
verified.

## 13. Monitoring and conservative stop criteria

Deduplicate `metrics.jsonl` by iteration with last-record-wins, then compare the
latest 100 completed iterations to the preceding non-overlapping 100 and to the
post-intervention pure-PPO baseline.

Monitor:

- total entropy and entropy coefficient
- action-kind, move-conditional, and build-conditional entropy
- underlying and behavior build probability
- build action share, builds per completed game, player-game build rate,
  eligibility rate, successful builds, and seat-specific rates
- total/policy/value loss, approximate KL, clip fraction, gradient norm,
  learning rate, explained variance, epochs used, reward/score
- episodes and W/L/D
- samples/second and iteration seconds
- counterfactual actor/critic total scales
- learned-league raw and EMA W/L/D, score, paired interval, macro score, castle
  statistics, and mean game length
- process health, GPU visibility, disk space, publication status, and W&B flow

For the current run, stop immediately for non-finite parameters/loss/gradients,
unrecoverable process failure, or disk pressure that threatens checkpoints.
Otherwise require confirmation across two independent 100-iteration windows or
two monitoring checks before stopping for:

- entropy below 0.20 and still falling
- mean KL above 0.03
- clip fraction above 0.25
- explained variance below 0.50
- castle collapse: below 0.75 builds/game **and** below 30% player-game build
  rate **and** below 0.1% build action share
- runaway building: above 10 builds/game **and** above 95% player-game build
  rate **and** above 1% build action share
- performance collapse: raw and EMA both decisively worse than the preceding
  corresponding checkpoint, or both macro scores below 0.45 at two consecutive
  scheduled evaluations

Do not stop for one noisy point, an entropy decline above the threshold, normal
castle-rate variation, throughput variation alone, or one weak raw/EMA result
when the other remains healthy.

## 14. Graceful stopping and cleanup

To stop early, resolve and validate the single
`continuation_supervisor` process for the exact run and send `SIGTERM` to that
supervisor only. Do not kill the training child directly. The supervisor
forwards the signal so training can finish its current operation, save
`terminal.eqx`, prepare publication, and run the terminal raw-versus-EMA
evaluation.

Before releasing the host:

1. Wait for the supervisor to exit cleanly and record its exit status.
2. Verify terminal checkpoint iteration, stage, SHA-256, and metadata.
3. Verify terminal raw-versus-EMA evaluation.
4. Upload every unmarked publication bundle and verify completion.
5. Preserve `metrics.jsonl`, standalone learned-league JSON files,
   `learned_league_manifest.json`, publication manifests/status, terminal
   evaluation, config, logs, and hashes in the repository run directory or
   another durable artifact bundle.
6. Verify the W&B run or deduplicated derived run reaches the terminal
   iteration.
7. Stop the GiveMeANode node or Google VM. Retain the disk only when it contains
   a still-needed recovery copy; otherwise remove it after remote verification.

Never leave an idle eight-H100 host billing because an upload or report failed.
Conversely, never stop/delete the host before the final checkpoint and compact
analysis artifacts are verified elsewhere.

## Launch acceptance checklist

- [ ] Exact source revision/archive recorded
- [ ] Newest durable full checkpoint selected, not merely newest W&B iteration
- [ ] Checkpoint SHA-256 and schema verified
- [ ] Python 3.11.15 and all four JAX components at 0.10.2
- [ ] Compatible CUDA/cuDNN/NCCL/`ptxas`
- [ ] Eight JAX CUDA devices visible
- [ ] Global batch geometry preserved
- [ ] Anchor plus every admitted learned-league checkpoint restored
- [ ] New run/config/HF paths are internally consistent
- [ ] W&B and Hugging Face credentials tested without exposing secrets
- [ ] Durable run storage mounted with adequate free space
- [ ] CPU supervisor has no GPU visibility; child has all eight GPUs
- [ ] First new iteration and W&B metrics confirmed
- [ ] Publication monitor active
- [ ] Conservative safety monitor active
- [ ] Graceful-stop and final-export procedure assigned before unattended run

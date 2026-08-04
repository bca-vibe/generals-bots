# Conv 1313 competition submission

Standalone export of the EMA policy from the final convolutional checkpoint:

- Hugging Face: `bca-vibe/generals-bot`, `runs/arch_ab_d448_8xh100_5h_retry1_20260803/branches/conv/terminal.eqx`
- Iteration: 1313
- Curriculum stage: 4
- Checkpoint SHA-256: `23cf2c76ea348ee67b3c0dd796920a03cc2a1268e641d323dd926a5803306bd2`
- Architecture: 7-layer 448-wide transformer with a 96-channel convolutional patch residual
- Observation schema: `competition_39`

The exported policy is bfloat16 and runs directly in JAX without Equinox or
the training repository. `build.sh` warms the persistent CPU compilation cache.

The upload zip contains `run.sh`, `build.sh`, `main.py`, `bot.py`,
`weights.npz`, `export_metadata.json`, and this README at its root.

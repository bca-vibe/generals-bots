# Smoke 1260 baseline submission

This bot runs the EMA policy from `smoke_8xh100` checkpoint iteration 1260.
The training checkpoint SHA-256 is
`878e7fb3964b82abc5f4f79a4335f19e3fbe0352500c962c200bc33d936225c5`.

The export stores policy parameters in bfloat16 and reimplements the original
`legacy_38` observation memory and transformer directly in JAX. Equinox and the
optimizer state are not required at competition runtime.

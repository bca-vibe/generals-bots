# Boss submission

Standalone competition export of `generals.agents.BossAgent`.

The policy is a deterministic NumPy port of the in-engine JAX heuristic. It
keeps the same tactical overrides, shortest-path routing, defensive screening,
castle economy, fog scouting, and deathtouch behavior without importing the
engine package or paying a JAX compilation cost.

The upload zip contains these files at its root:

- `run.sh`
- `main.py`
- `agent.py`

No build step or vendored dependencies are required. NumPy is included in the
official competition runtime.

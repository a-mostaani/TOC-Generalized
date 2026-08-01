# TOC-Generalized

A JAX/Python port of **SAIC** (State Aggregation for Information
Compression), a task-oriented-communication algorithm: agents doing
multi-agent RL under a rate-limited channel learn to compress what they
observe into a few bits before sharing it, via value-based state
aggregation. The environment is a geometric-consensus grid-world
rendezvous task (agents must reach a shared goal cell simultaneously).
The port also incorporates **ESAIC**'s Theorem 1: the centralized
(full-observability) training phase only ever needs to run with 2 agents,
regardless of how many agents the eventual decentralized policy is
trained for.

Ported from the original MATLAB implementation (`SAIC/`, `Fully
Centralized - MultiAgent/`), validated against real Octave-executed
MATLAB reference output (`validate/compare_to_matlab.py`).

## Pipeline

```
1. Centralized training (noa=2, full observability)   -> jax_saic/centralized.py
2. Value-of-observation + k-median clustering           -> jax_saic/clustering.py
   (compresses each agent's raw position into inf_bits of "cluster id" --
   this compression IS the learned communication policy)
3. Decentralized training (noa=N, agents only exchange  -> jax_saic/train.py
   the compressed cluster id, not raw position)
```

## Getting started

```
python -m venv .venv
source .venv/bin/activate
pip install jax jaxlib "numpy<2" matplotlib scipy
python validate/compare_to_matlab.py
```

## Documentation

- **`CLAUDE.md`** — orientation: module layout, MATLAB-to-Python
  conventions, non-goals.
- **`PORT_NOTES.md`** — the source of truth for every semantic detail,
  MATLAB quirk, hyperparameter, RNG convention, and open question. Read
  this before changing behavior in `jax_saic/`.

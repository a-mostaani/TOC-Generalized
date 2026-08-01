# CLAUDE.md — SAIC MATLAB → JAX port

This file orients a future Claude session (or human) to this repo quickly.
For every semantic detail, MATLAB quirk, hyperparameter, RNG convention, or
open question, **`PORT_NOTES.md` is the source of truth** — read it before
changing behavior in `jax_saic/`. This file only maps the terrain.

## What this is

A JAX/Python port of SAIC (State Aggregation for Information Compression),
a task-oriented-communication algorithm: agents doing multi-agent RL under
a rate-limited channel learn to compress what they observe into a few bits
before sharing it, via value-based state aggregation. The environment is a
geometric-consensus grid-world rendezvous task (agents must reach a shared
goal cell simultaneously). The port also incorporates ESAIC's Theorem 1:
the centralized (full-observability) training phase only ever needs to run
with 2 agents (`NOA=2`), regardless of how many agents (`N`) the eventual
decentralized policy is trained for — see `PORT_NOTES.md` §0.8/§4.4.

The three-phase algorithm:

```
1. Centralized training (noa=2, full observability)  -> centralized.py
2. Value-of-observation + k-median clustering          -> clustering.py
   (compresses each agent's raw position into inf_bits of "cluster id" —
   this compression IS the learned communication policy)
3. Decentralized training (noa=N, agents only exchange  -> train.py
   the compressed cluster id, not raw position)
```

## Module layout (`jax_saic/`)

| File | MATLAB source (see PORT_NOTES.md) | What it does |
|---|---|---|
| `indexing.py` | (shared helper, no direct MATLAB equivalent) | `flatten_mixed_radix`/`unflatten_mixed_radix`: 0-indexed little-endian mixed-radix encode/decode, used for BOTH joint-state/action flattening and bit-message encoding. Equivalent to MATLAB's `de2bi`/`bi2de` under 'right-msb', but 0-indexed throughout instead of MATLAB's 1-indexed tables. |
| `env.py` | `envir_gc.m` | `step(ps0, pa0, n, goal_set0)` — jitted grid-world transition. `at_goal()` helper. Action constants `RIGHT,LEFT,UP,DOWN,STAY = 0,1,2,3,4`. Includes the goal_set membership-test fix (§2). |
| `channel.py` | `bsc_ch.m` | Standalone BSC channel, generalized past MATLAB's noa=2-only bug (§3). **Not wired into training** — the pipeline models a rate-limited-but-otherwise-perfect channel (the bit budget is the compression `inf_bits`, not channel noise). |
| `centralized.py` | `benchmark_perfectcom_MultiAgent.m` + `bench_policy_UCB.m` + `pbench_update.m` | `train(cfg, rng)` — centralized Q-learning at `NOA=2` (module-level constant). Returns `(qp_table, N_table_emerged, rew)`. `CentralizedConfig` holds hyperparameters (§8.1). |
| `clustering.py` | `sum_q_MultiAgent.m` (→ `value_of_observation`) + `aggregate_states_SAIC.m` (→ `cluster_states`) | Value-of-observation via empirical, visitation-weighted marginalization over the other agent's position, then k-median/k-medoids state aggregation into `inf_bits` clusters. Three `method=` options: `"kmedian"` (default, exact DP), `"kmedoids"` (PAM heuristic), `"legacy_minus50"` (faithful replica of MATLAB's original crash-prone bug, for comparison only). |
| `qlearning.py` | `ppolicy_customized_nbits_UCB_bestrew_ma.m` + `pupdate_customized_nbits_ma.m` | `select_action()` / `update()` for the decentralized phase — SARSA-style tabular Q-learning over the compressed (clustered) joint observation. |
| `train.py` | `EoC_SAIC_3Agents.m` | `DecentralizedConfig`, `s_aggregate()` (raw position → cluster id via `ag_states`), `train(cfg, ag_states, rng)`, and `run_phase1(...)` — the full 3-phase orchestration entry point. |

All RNG is explicit `jax.random.PRNGKey` threading — no global RNG state
anywhere, unlike MATLAB's `rng(seed)`.

## Validation (`validate/`)

`compare_to_matlab.py` is the main entry point: runs the JAX pipeline
across seeds/`noa` values and overlays it against real Octave-executed
MATLAB reference output (`octave_work/`, `octave_shims/` — Octave used as
a MATLAB substitute, with several genuine Octave/MATLAB compatibility
gaps patched via minimal shims, not behavior changes — see §11). Produces
`reward_vs_episode.png` and `reward_vs_noa.png` in `validate/plots/`.

Everything else in `validate/` is a diagnostic built in response to a
specific question raised during validation, not part of the pipeline:

- `optimal_return.py` — closed-form exact optimal expected return (no RL),
  used to normalize measured returns against a true ceiling (§11.2).
- `check_central_return.py`, `sweep_central_length.py`, `sweep_add_400k.py`
  — is the centralized phase itself converging, and does training longer
  help clustering (§11.1/§11.3)?
- `check_value_precision.py` — do learned Q-values match exact value
  iteration on the known MDP (§11.4)? (Yes, to 4 decimals.)
- `plot_grid_clusters.py` / `plot_grid_clusters_exact.py` — Fig-8-style
  (SAIC paper) grid cluster visualization, empirical vs. exact
  value-of-observation.
- `exact_value_of_observation.py` — a fully deterministic, zero-variance
  alternative to `clustering.value_of_observation()`, built after the
  empirical version showed real (non-noise-shrinking) asymmetry between
  grid-symmetric states (§11.5). **Diagnostic only, by explicit decision**
  — not wired into `clustering.py`; see §11.5 for the integration point if
  this is revisited.
- `check_symmetry_multiseed.py`, `compare_anchored_ag_states.py` —
  seed-variance and clustering-vs-decentralized-training ablations.

Run any of these with the repo's `.venv` (`jax`, `jaxlib`, `numpy<2`,
`matplotlib`, `scipy` — do NOT install JAX into a shared/base conda
environment; see §11 intro for why).

## MATLAB → Python conventions to know before touching this code

- **Indexing**: MATLAB tables are 1-indexed; every JAX array here is
  0-indexed. Any place a MATLAB comment says "state 5" corresponds to
  index 4 here. `PORT_NOTES.md` documents the mapping at each call site
  it matters.
- **Mixed-radix flattening**: joint states/actions and bit-messages both
  use `indexing.py`'s little-endian mixed-radix helpers — same math as
  MATLAB's `de2bi`/`bi2de`('right-msb'), just 0-indexed.
  `agent0_idx + n2*agent1_idx` = "agent-0-fastest" convention, used
  consistently in `clustering.py`/`centralized.py`.
- **Centralized training is always `noa=2`** regardless of the
  decentralized target `N` (ESAIC Theorem 1) — don't "fix" `centralized.py`
  to accept other `noa` values; that's intentional, not an oversight.

## Non-goals (stated at project start, still in force)

- No environments/channels/algorithms beyond this one rendezvous task and
  this one SAIC/ESAIC pipeline.
- No framework integration (no Flax/Haiku/etc. — plain JAX).
- No neural networks — tabular Q-tables only, matching MATLAB exactly.
- No performance optimization beyond whatever `jax.jit` gives for free.
- **Never "fix" or "improve" MATLAB behavior found to be odd, buggy, or
  suboptimal without asking first.** Every deviation from literal MATLAB
  behavior in this repo was explicitly discussed and approved — flag
  anything new the same way, in `PORT_NOTES.md`, before changing it.

## Where the original MATLAB lives

- `SAIC/` — the broader, mostly-unused MATLAB project (many variant
  scripts; only `envir_gc.m`, `bsc_ch.m`, and `EoC_SAIC_3Agents.m` are the
  actual reference driver — see `PORT_NOTES.md` §0 for how this was
  determined).
- `Fully Centralized - MultiAgent/` — `benchmark_perfectcom_MultiAgent.m`,
  `aggregate_states_SAIC.m`, `sum_q_MultiAgent.m`, etc. — the centralized
  training + clustering reference (ported by `centralized.py`/`clustering.py`).
- `ProblemVarients2018Until2023/` — historical variants, not otherwise
  used by this port.

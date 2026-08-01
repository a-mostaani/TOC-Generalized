# PORT_NOTES.md — SAIC MATLAB → JAX, Phase 1

Source repo: `https://github.com/a-mostaani/SAIC` (cloned to `./SAIC`).
Paper 1 (SAIC): A. Mostaani, T. X. Vu, S. Chatzinotas, B. Ottersten, "Task-Oriented
Data Compression for Multi-Agent Communications Over Bit-Budgeted Channels,"
IEEE OJ-COMS vol. 3, 2022. arXiv:2005.14220. Equation/theorem numbers for this
paper refer to the arXiv PDF.
Paper 2 (ESAIC): A. Mostaani, T. X. Vu, H. Habibi, S. Chatzinotas, B. Ottersten,
"Task-Oriented Communication Design at Scale," IEEE Trans. Communications,
vol. 73, no. 1, Jan 2025 (`Task-Oriented_Communication_Design_at_Scale (3).pdf`,
added to the project root — see §0.8). Equation/theorem numbers for this paper
are prefixed `ESAIC` below to disambiguate from Paper 1's numbering.

This document is the record of Step 1 (read-only analysis). Nothing here has
been implemented yet. **Several scope decisions were made interactively with
the repo owner during this step; they are recorded in §0 and supersede the
original task's file list.** Everything downstream (§4 in particular) still
has open questions — see §9.

---

## 0. Scope resolution (read this first)

The task originally named six files as Phase 1's scope: `envir_gc.m`,
`bsc_ch.m`, `cpolicy_customized_nbits.m`, `cupdate_customized_nbits.m`,
`ppolicy_customized_nbits.m`, `pupdate_customized_nbits.m` (with `EoC_SAIC.m`
as the driver tying them together). Reading the code surfaced a hard
contradiction: **no single script in the repo actually wires those functions
together.**

- `EoC_SAIC.m` never calls `cpolicy_customized_nbits`, `cupdate_customized_nbits`,
  or `ppolicy_customized_nbits`. It calls `ppolicy_customized_nbits_UCB_bestrew`
  instead, derives its communication signal deterministically from a
  precomputed state-aggregation table (`ag_states_median`) rather than a
  learned Q-table, and depends on a `load(...)` call for that table which is
  commented out — the script cannot run as committed.
- `pupdate_customized_nbits.m` doesn't exist as a file; only `_2`, `_gray`,
  `_ma` variants do.
- The only script that calls the bare `cpolicy_customized_nbits` /
  `cupdate_customized_nbits` / `ppolicy_customized_nbits` together is
  `EoC_par_function.m`, which uses `envir.m`, not `envir_gc.m`.

Resolution, decided with the repo owner (who is the paper's author):

1. **Reference driver is `EoC_SAIC_3Agents.m` + `envir_gc.m`**, not `EoC_SAIC.m`
   and not `EoC_par_function.m`. This driver is itself generic over agent
   count `noa` (not hardcoded to 3 despite the filename) and that generality
   is kept, since it's intrinsic to the one algorithm variant being ported,
   not a new abstraction.
2. **No learned communication Q-table in Phase 1.** `cpolicy_customized_nbits.m`
   / `cupdate_customized_nbits.m` are dropped from scope. Communication content
   is the output of the SAIC state-aggregation/clustering step (see §4), which
   the repo owner confirmed *is* the "learned communication policy" — it's
   learned offline (once, before the RL loop), not online via a per-step
   Q-table.
3. **Position policy/update use the `_ma` / `_UCB_bestrew_ma` variants**
   (`ppolicy_customized_nbits_UCB_bestrew_ma.m`, `pupdate_customized_nbits_ma.m`),
   since those are what `EoC_SAIC_3Agents.m` actually calls, not the plainer
   `ppolicy_customized_nbits.m` / `pupdate_customized_nbits_2.m`.
4. **The `agreggated_states_n3_g9_infbits2_realSAIC.mat` dependency has no
   source anywhere in the SAIC repo or its git history.** It is the *output*
   of a separate phase of SAIC — a centralized training phase plus a
   k-median clustering step (Algorithm 1, steps 1–7 of the paper). At the
   time this decision was made, no MATLAB source for that phase was known to
   exist, so the plan was to implement it fresh from the paper's equations.
   **Superseded by item 7 below** — that source turned out to exist after
   all, just not in the SAIC repo itself.
5. **Channel model: rate-limited but perfect (`bits == inf_bits`).** This is
   the intended model for Phase 1, not a simplifying compromise: the
   channel's only constraint is *how many bits it can carry per channel
   use* (its rate). The communication design is assumed to already respect
   that budget — no message is ever encoded with more bits than the
   channel's rate limit — and once respected, transmission is perfect
   (error-free). Concretely, this is exactly `EoC_SAIC_3Agents.m`'s
   `bits == inf_bits` path: no noise, `bsc_ch` never invoked, `bsc_p` has no
   effect. The `bits > inf_bits` path (redundant coding to protect against
   an *unreliable* channel, requiring `bsc_ch.m` plus MATLAB Communications
   Toolbox cyclic-code `encode`/`decode`) models a different problem — a
   channel that can drop/flip bits even within its rate limit — which is
   out of scope for Phase 1. `bsc_ch.m` is still ported and unit-tested as
   a standalone function (it was explicitly named in scope, and a later
   phase may want the noisy-channel model) but is **not wired into the
   Phase 1 training loop**, by design.
6. **`p(o₋ᵢ(t))` in eq. 15/43 is estimated empirically from centralized-
   training rollout visitation frequency**, not assumed uniform.
7. **Update, post-hoc:** after this document's first draft, the repo owner
   added two folders to the working tree — `Updated in Luxembourg/` (a
   large collection of adjacent MATLAB projects) and `Fully Centralized -
   MultiAgent/` (extracted from the former, placed at the project root).
   **This folder contains the actual MATLAB source for the phase §4
   originally had to derive from the paper alone.** It is not a from-scratch
   implementation task anymore — it's a port, like everything else in this
   document. §4 below is rewritten against that source. Sibling folders
   `Fully Centralized - UCB/` (2-agent-only, no clustering code) and `Fully
   Centralized - Deep RL MultiAgent/` (neural-network value function) were
   checked and are not used — the former is superseded by the general-`noa`
   version here, the latter conflicts with the task's explicit "no neural
   network" non-goal.
8. **Update, post-hoc #2 — resolves §9 item 1.** The repo owner added a
   second paper to the project root, "Task-Oriented Communication Design at
   Scale" (ESAIC — Extended SAIC), and explained its central claim: for a
   *symmetric* multi-agent system, the centralized-training + value +
   clustering phase (§4) does **not** need to be run with the target
   decentralized agent count `N`. It can be run once with `noa=2`, and the
   resulting `ag_states_median` reused for decentralized training at any
   `N`. This is exactly ESAIC's Theorem 1 / Algorithm 1 (ESAIC paper,
   Section IV): centralized training solves problem (9) for a **two-agent**
   system to get `V*[2](·)`, the clustering problem (10)/(11) is solved
   using that 2-agent value function, and only the *decentralized* phase
   (Section IV-C of the ESAIC paper) runs at the real `N`. The paper's own
   Theorem 1 gives a sufficient condition (its eq. 13, condition `c1`) under
   which this produces the *same* partition/communication-policy that
   running the full `N`-agent centralized training would have — i.e. it's
   not a heuristic shortcut, it's the paper's actual proven result.
   **Consequence:** `benchmark_perfectcom_MultiAgent.m`'s own `noa=4`
   default (§4) was for the ESAIC paper's *own numerical validation* of
   Theorem 1 — comparing SAIC-style (`N`-agent-trained) VoI against
   ESAIC-style (2-agent-trained) VoI at higher `N` to show they coincide —
   not the production recipe for feeding `EoC_SAIC_3Agents.m`. Decision:
   **Phase 1's centralized training (§4) runs at `noa=2`, unconditionally,
   regardless of the decentralized phase's agent count.** This sidesteps
   §9's former item 1 (the `decod_N_table` `noa≥4` blocker) entirely —
   that code path is fully correct at `noa=2`, and Phase 1 never needs to
   call it at `noa≥4` in the first place. See revised §4.4 and §9.
   One further simplification, tied to the ESAIC paper's `Nᵢᶜ`/`Bᵢ,ⱼ`
   machinery (its eq. 8/10): a general `N`-agent system may need one
   distinct clustering per *distinct bit-budget* an agent uses across its
   different neighbors. Phase 1's rendezvous grid-world is symmetric with a
   single, uniform `inf_bits` for every agent pair, so exactly **one**
   clustering/`ag_states_median` is needed regardless of `N` — matching
   what `EoC_SAIC_3Agents.m` already does (one shared table for all
   agents). Flagging this as a real simplifying assumption specific to this
   environment's symmetry, not a general truth — worth remembering if a
   later phase introduces heterogeneous bit-budgets.

Net effect: Phase 1 now spans the *entire* SAIC Algorithm 1 (all three
phases — centralized training, clustering, decentralized training), not a
six-function slice of just the decentralized part. That's a materially
larger scope than the original task framing ("ONE environment + ONE
algorithm variant... six files"), reached deliberately through the
conversation above, not assumed unilaterally. Unlike the original six
files, every piece of it now has real MATLAB source behind it.

---

## 1. File map for this phase

| Role | File | Status |
|---|---|---|
| Environment | `envir_gc.m` | Port as-is (identical copy also present in `Fully Centralized - MultiAgent/`) |
| Channel | `bsc_ch.m` | Port as standalone unit; not exercised by the main loop (see §0.5) |
| Decentralized training driver | `EoC_SAIC_3Agents.m` | Port its actual logic (not `EoC_SAIC.m`) |
| Position policy | `ppolicy_customized_nbits_UCB_bestrew_ma.m` | Port as-is |
| Position update | `pupdate_customized_nbits_ma.m` | Port as-is |
| Centralized training driver | `Fully Centralized - MultiAgent/benchmark_perfectcom_MultiAgent.m` | Port its actual logic — §4 |
| Centralized policy | `Fully Centralized - MultiAgent/bench_policy_UCB.m` | Port as-is |
| Centralized update | `Fully Centralized - MultiAgent/pbench_update.m` | Port as-is |
| Value computation | `Fully Centralized - MultiAgent/sum_q_MultiAgent.m` | Port as-is |
| Clustering | `Fully Centralized - MultiAgent/aggregate_states_SAIC.m` | Port as-is (MATLAB `kmedoids` — needs an equivalent, see §9) |
| Dropped | `cpolicy_customized_nbits.m`, `cupdate_customized_nbits.m`, `pupdate_customized_nbits.m`/`_2`/`_gray`, `ppolicy_customized_nbits.m` | Out of scope per §0.2/§0.3 |
| Not used | `Fully Centralized - MultiAgent/sum_q.m`, `benchmark_perfectcom_odd.m`, `benchmark_perfectcom_UCB_goal_changing*.m`, `benchmark_UCB_goal_changing_bestrew_updated_costlymoves.m` | Siblings in the same folder for other scenarios (2-agent-only value fn, goal-changing variants, "odd" variant) — not called by the path documented in §4, not read in depth |

Hyperparameter defaults for the decentralized phase are taken from
`parallel_simulator_2phase_encoded_MA.m`, the only script in the (main SAIC)
repo that calls `EoC_SAIC_3Agents` with concrete literal values (for
`noa = 4`; comments in the same file give `ns` for `noa = 2` and `noa = 3`).
Defaults for the centralized phase are taken from
`benchmark_perfectcom_MultiAgent.m` itself, in §8.

---

## 2. Environment — `envir_gc.m`

Plain English: an `n × n` toroidal-free grid. Each agent occupies one cell,
numbered `1..n²` in column-major order (MATLAB convention: cell index
increases down a column first, then across columns). Agents pick one of 5
actions per step: `1`=right, `2`=left, `3`=up, `4`=down, `5`=stay. Moving off
an edge is illegal — the action becomes a no-op and `err(i)=1` is set (but
`err` is discarded by every caller; it's dead output). One designated cell,
`goal_set`, is the rendezvous point.

State/action semantics:
- Position state `ps(i) ∈ {1, ..., n²}`, 1-indexed.
- Position action `pa(i) ∈ {1,2,3,4,5}`.
- Coordinate decoding: `y = fix((ps-0.05)/n) + 1`, `x = rem(ps,n)` (with `x=0 → x=n`).
  The `-0.05` is a float-rounding guard for `fix()` on exact multiples of `n`.
  Converting back: `ps' = x' + n·(y'-1)`.
- Terminal signal `ter` is a **count**, not a boolean: `ter` increments once
  per agent whose new `ps(i)` equals `goal_set`. So `ter ∈ {0, ..., noa}`.

**Fixed, per explicit instruction (deviation from MATLAB, deliberate and
confirmed — not a silent fix):** `if ps(i) == goal_set` performs elementwise
MATLAB `==`, which only behaves as "is ps(i) in the goal set" when
`goal_set` is a single-element vector (as it is in every driver seen, e.g.
`goal_set=[9]`). For a genuine multi-cell `goal_set`, `if` on a non-scalar
result implicitly requires *all* elements to be true (MATLAB `if` calls
`all()`), silently breaking the terminal check. You confirmed the intended
fix: **check membership** — `any(ps(i) == goal_set)` (or equivalently
`ismember(ps(i), goal_set)`) — instead of raw `==`. The JAX port's `env.py`
will implement `ter` accumulation this way, so a real multi-cell `goal_set`
works correctly, not just the single-cell case every existing MATLAB driver
happens to use.

**Consistency note:** the same `ps == goal_set` pattern (same bug) also
appears three more times in `EoC_SAIC_3Agents.m` — the action-canceling
check (`if ps(j)==[goal_set]`, line 273) and both `rew_winner` membership
checks (`if ps(ii)==goal_set`, lines 358/367). Fixing only `envir_gc.m`
while leaving those as raw `==` would half-defeat the point: `ter` would
correctly count arrivals at any goal cell, but the training loop wrapping
it would still only recognize a single specific cell for action-canceling
and reward attribution. Applying the same `any(...)`/membership fix
consistently across all three in the port (§5), so `goal_set` genuinely
behaves as a set everywhere it's used. `benchmark_perfectcom_MultiAgent.m`
(§4.1) already does this correctly as written (`sum(ps==goal_set)>=1`,
which broadcasts over a multi-element `goal_set` correctly) — no fix needed
there.

**Backward compatible by construction:** at `goal_set=[g]` (singleton — what
every existing MATLAB driver actually uses), `any(ps(i)==goal_set)` reduces
to plain `ps(i)==g`, identical to the original behavior. The membership-test
version is a strict superset — same result on every config that exists
today, plus correct behavior for a genuine multi-cell `goal_set`, with no
separate code path needed for either case.

No RNG consumed inside `envir_gc.m`.

Index mapping to carry into JAX (document in code comments per your
instruction re: 1-indexing off-by-ones): MATLAB `ps ∈ {1..n²}` → JAX will
almost certainly want `ps0 = ps - 1 ∈ {0..n²-1}` internally for array
indexing, with the MATLAB row/col decoding formulas re-derived for 0-index.
This translation is the single highest-risk spot for an off-by-one; the
validation script should specifically check terminal-state detection and
edge-collision behavior cell-by-cell against MATLAB before trusting anything
downstream.

---

## 3. Channel — `bsc_ch.m`

Plain English: a memoryless binary symmetric channel. Given `cs_bc` — the
bits agent `i` is about to receive from every other agent, shape
`(noa, noa-1, bits)` — each bit is independently flipped with probability
`bsc_p`.

```
for i in agents, j in bits:
    if rand() <= 1 - bsc_p:  cs(i,1,j) = cs_bc(i,1,j)   # correct
    else:                     cs(i,1,j) = ~cs_bc(i,1,j)  # flipped
```

RNG consumed: one `rand()` draw per `(agent, bit)` pair, `noa × bits` draws
total per call, in row-major (agent-outer, bit-inner) order.

**Note:** per §0.5, Phase 1's channel model is rate-limited but perfect —
the inter-agent communication design is assumed to already respect the
channel's bit-rate budget, and no bit errors occur once that budget is
respected. `bsc_ch.m` models a *different, unreliable* channel (bit flips
even within budget), which is a distinct problem outside Phase 1's scope.
It's still ported and unit-tested standalone (verify empirical flip rate ≈
`bsc_p` over many trials, verify `bsc_p=0` is a no-op, `bsc_p=1` is a full
inversion) for later phases, but is not called from `train.py`'s main loop
— the reference driver's rate-limited-perfect branch doesn't call it
either.

---

## 4. Centralized training + clustering — `Fully Centralized - MultiAgent/`

Per §0.7, this is now a **port**, not new work. Source: `benchmark_perfectcom_MultiAgent.m`
(driver) + `bench_policy_UCB.m` (centralized action policy) +
`pbench_update.m` (centralized Q-update) + `sum_q_MultiAgent.m` (value-of-
observation) + `aggregate_states_SAIC.m` (clustering). This is what actually
produces the `ag_states_median` artifact `EoC_SAIC_3Agents.m` loads — the
correspondence is exact and checkable: `benchmark_perfectcom_MultiAgent.m`
line 387 builds `batch_ag_states_median(:, :, b)` for each of `bn` batches
via `aggregate_states_SAIC`, and `EoC_SAIC_3Agents.m` line 36 consumes
`batch_ag_states_median(:,:,1)` — the first batch — from a `.mat` file with
exactly that variable name.

### 4.1 Centralized training (`benchmark_perfectcom_MultiAgent.m` + `bench_policy_UCB.m` + `pbench_update.m`)

The joint state of all `noa` agents is flattened into one scalar index and
treated as a single-agent MDP over a `(n²)^noa`-state, `5^noa`-action table
— this is eq. 3 of the paper, and it's exactly what the code does:

```
main_ps = ps(1) + Σ_{k=2}^{noa} (ps(k)-1)·(n²)^(k-1)     # mixed-radix, base n², 1-indexed
main_pa = pa(1) + Σ_{k=2}^{noa} (pa(k)-1)·5^(k-1)         # same scheme, base 5
```
(`ps_calc`/`pa_calc` invert this; `mps_calc` is the same forward map restated.)

`qp_table` is `(n²)^noa × 5^noa`, init `0.02`. `N_table`/`N_table_emerged`
same shape, init `0.001` (visitation counts, smoothed to avoid `log(0)` in
the UCB bonus — same trick as the decentralized phase's `NE_table`).
`N_table_emerged` only accumulates once `episode > end_learn·ns`, exactly
mirroring the decentralized phase's post-`end_learn` counter split.

**Policy** (`bench_policy_UCB.m`): `policy='ep-greedy'` is hardcoded at the
top of the function (line 19) — in MATLAB, `policy` isn't even a parameter
of this function, so every other branch (`'ucb'`, `'stochastic'`,
`'stochastic-ucb'`, `'stochastic-epsilon'`) is unreachable dead code.

**RESOLVED (§9 item 7): real, user-facing `centralized_policy` parameter**,
independent of the decentralized phase's `decentralized_policy` (§6) since
the option sets don't overlap:
- `"ep-greedy"` (default, matches every existing MATLAB run):
  ```
  epsilon = 1 - episode / (end_learn · ns)      # anneals linearly to exactly 0, no floor
  if rand() < epsilon:  main_pa = randi(5^noa)  # ONE draw over the flattened joint action
  else:                 main_pa = argmax_a qp_table(main_ps, a)
  ```
  Note this draws **one** uniform sample over all `5^noa` joint actions when
  exploring — not `noa` independent per-agent draws. Different epsilon
  formula from the decentralized phase's `_ma` policy (§6.1: floors at
  `0.02`, not `0`) — two distinct, independently-tuned annealing schedules;
  don't conflate them.
- `"ucb"` — deterministic `argmax` of `Q + UCB-bonus` over the flattened
  joint-action table, only during the `update` window (`episode <
  ns·end_learn`); falls back to plain greedy afterward.
- `"stochastic-ucb"` — Boltzmann/softmax over `(Q + UCB-bonus)/tau`,
  sampled by cumulative probability.
- `"stochastic-epsilon"` — adaptive switch between softmax and greedy based
  on a value-difference criterion (Tokic 2011).
- `"stochastic"` — Boltzmann/softmax over `Q/tau`. **MATLAB hardcodes
  2-agent indexing here** (`qp_table(ps(1),ps(2),:,:)`), inconsistent with
  every other branch's flattened joint-state index — broken for `noa≠2` as
  written. **Fixed in the port** (per your instruction) to use the same
  flattened `qp_table(main_ps, :)` indexing as the other branches, so it
  works generally rather than only by accident at `noa=2`. This is a
  behavior change from literal MATLAB, scoped to this one branch, done
  because you asked for it rather than silently.

All five are real, selectable, working options in the port — none are
present-but-unreachable.

**Update** (`pbench_update.m`): standard off-policy Q-learning (bootstraps
via `max` over the *next* joint state's full action row — not SARSA, unlike
§7's decentralized update), `alpha=0.07`, `gamma=0.9` both hardcoded inside
the function (again not threaded from the caller's `gamma` argument — same
pattern flagged in §7).
```
if temp_rew == 0:  qp_table[last_ps, pa] += alpha·(gamma·max_a qp_table[ps,a] - qp_table[last_ps,pa])
else:              qp_table[last_ps, pa] += alpha·(temp_rew - qp_table[last_ps,pa])
```

**Reward in the centralized driver** — `temp_rew = best_rew` (flat, no
exponent) when `ter==noa`, `temp_rew = 1` when `ter>=1` partial. **This is
by design, not evidence of a bug elsewhere** — see the correction in §5.3.9:
the centralized phase computes `V*[2]` from a single joint policy that
explores the *entire* joint action space directly, so it doesn't face the
combinatorial-rarity problem that makes independent decentralized agents
need reward shaping to find simultaneous arrival at all. This flat
reward is exactly eq. 15 of the ESAIC paper (§5.3.9): `r = C2` (full
simultaneous arrival), `C1` (partial), `0` (otherwise), with `C1 < C2` —
here `C2 = best_rew`, `C1 = 1`. The centralized phase's reward and the
decentralized phase's `temp_rew` are two different signals for two
different purposes, not two implementations of the same formula — see
§5.3.9 for the full explanation of why they legitimately differ.

RNG consumed: `randi(n²-1, noa, 1)` for initial positions (per episode) plus,
per step, one `rand()` for the epsilon check and — only if exploring — one
more `randi(5^noa)` (a single draw, not per-agent).

Episode/batch structure: MATLAB's outer loop runs `bn=5` independent
batches, each running the full `ns`-episode centralized training from a
fresh Q-table. **RESOLVED (§9 item 9): Phase 1 runs a single batch, not
5** — confirmed with the repo owner that one batch is sufficient to produce
the aggregated states; `EoC_SAIC_3Agents.m` only ever consumed batch 1 of
MATLAB's 5 anyway (§4.4), so this matches what was actually used downstream,
just without paying for the other 4 unused batches. `bn=5` in the MATLAB
script was presumably there to eyeball run-to-run variance during
development, not because the pipeline needs multiple batches.

### 4.2 Value of an observation (`sum_q_MultiAgent.m`)

This is the concrete, empirical realization of eq. 15/43 — resolves §0.6/
former-§9-item-1 precisely, and it is *not* a plain visitation-frequency
average; read carefully:

```
N_table = decode(N_table_emerged)              # unflatten joint index -> per-agent dims + action dim
prob_table = max(N_table, axis=action) / sum(N_table_emerged)   # see note below
qp_table_full = decode(qp_table)
v_o_mat = max(qp_table_full, axis=action)      # = V*(joint state) = max_m Q*(s,m), eq. 42
v_o_weighted = v_o_mat * prob_table
V_o_1[i] = sum over all OTHER agents' positions of v_o_weighted[..., agent_noa_pos=i]
N_o_1[i] = sum over all OTHER agents' positions AND the action axis of N_table[..., agent_noa_pos=i]
```

**Important approximation, verbatim from the MATLAB comment:** `prob_table`
uses `max` over the action axis of the visitation-count table, not `sum` —
i.e. it approximates "how often was this joint state visited" by "how often
was this joint state's *most-taken* action taken," on the stated assumption
that post-convergence the greedy policy only ever takes one action per
state ("since we are evaluating the ~optimal policy, other actions should
not occur"). This is the actual empirical-distribution estimator for
`p(o₋ᵢ)` — port it exactly as this heuristic, not as a naive `sum`-based
visitation frequency (which would double-count multiply-visited
state-action pairs differently). `V_o_1`/`N_o_1` are computed **only for the
last agent index (`noa`)** — by the rendezvous problem's symmetry, every
agent's observation space and value profile are identical, so one
computation is reused for all agents. Confirm this symmetry assumption
still holds before reusing this shortcut in a differently-shaped
environment (out of scope for Phase 1, noted for later-phase awareness).

`decod_N_table`'s inner unflattening (`decod_N_table` local function) is
only explicitly implemented for `noa ∈ {2, 3}` — it emits a `warning` and
silently produces wrong output for `noa ≥ 4` (falls through to the `else`
branch, which does nothing, leaving `N_table` all-zero). **RESOLVED, not a
blocker (§0.8):** per ESAIC's Theorem 1, §4's centralized training/value/
clustering pipeline always runs at `noa=2` regardless of the decentralized
phase's target agent count (§4.4) — so this function never needs to be
called above `noa=3` in Phase 1's actual pipeline in the first place.
`benchmark_perfectcom_MultiAgent.m`'s own `noa=4` default is the ESAIC
paper's separate scalability-validation experiment (§0.8/§4.4), not
something Phase 1 needs to reproduce or fix.

### 4.3 Clustering (`aggregate_states_SAIC.m`)

```
V_weighted = replicate each V_o_1[i] exactly floor(N_o_1[i]) times   # empirical resampling by visitation count
medoids = kmedoids(V_weighted, 2^inf_bits, distance='euclidean')     # returns medoid VALUES, not just labels
for each of the n² DISTINCT raw states i:
    agr_st[i] = argmin_k |V_o_1[i] - medoids[k]|                    # nearest-medoid assignment, direct
ag_states_median[cluster_id, :] = list of raw states assigned to that cluster (zero-padded)
```

**Resolved — verified correct, then simplified (§9 item 3).** Two questions
were checked against the SAIC paper before deciding anything here:

1. *Is replicating values by visitation count a legitimate way to solve
   eq. 13?* Yes. Eq. 12 (which eq. 13 approximates) is an expectation over
   the trajectory-induced distribution of observations, and Appendix B
   states explicitly that this expectation "can be estimated by computing
   it over the **empirical distribution**." Replicating each value by its
   (integer) visitation count and running an unweighted median/medoid
   solver on the result is not an approximation of a frequency-weighted
   median — it's an *exact* reduction of one to the other. This is the
   standard technique for feeding a weighted problem into an unweighted
   solver, and it's exactly what the paper's own appendix calls for.
2. *Is `kmedoids('Distance','euclidean')` solving eq. 13's actual problem?*
   Yes. Eq. 13's cost is `Σ|V*(o)-μ'_k|` (L1/absolute-difference). For
   **scalar (1-D) data**, Euclidean distance between two numbers is just
   their absolute difference — the two costs are identical in one
   dimension, not merely similar. And for an L1 cost, a cluster's optimal
   center (its median) is always one of its own points for a finite set —
   so "medoid" and "median" coincide here too. What differs between
   `kmedoids` and an exact solver is **algorithm** (PAM local search vs.
   exact global optimum), not **problem** — confirming the `"kmedoids"` vs
   `"kmedian"` runtime-option split below is the right framing.

`kmedoids` was simply what MATLAB had built in at the time this code was
written (confirmed with the repo owner) — not chosen because it was *the*
correct method, just an available approximation to it. **Decision: make
the clustering method a runtime option**, not a single hardcoded choice —
the JAX port's clustering function takes a `method` argument selecting
between:
- `"kmedoids"` — a PAM/k-medoids heuristic, replicating this MATLAB
  behavior for fidelity/validation against the original MATLAB output.
- `"kmedian"` (default) — an exact solve of eq. 13's actual problem. Since
  the values are 1-D scalars, this has an efficient exact DP solution
  (sort the values, DP over contiguous partitions) — no approximation
  needed here, unlike `kmedoids`.

Both options should be exposed through `train.py`'s / the validation
script's config, so a run can be pointed at either implementation instead
of committing to one at the port's design time.

**The `-50` magic-number offset is removed, not ported.** The original
MATLAB reads a cluster label back out of the *replicated* array at index
`sum_no[i]-50`, reasoning that samples for state `i` occupy contiguous
indices `sum_no[i-1]+1..sum_no[i]`, all with the identical value, so any
one sample 50 positions inside that block (away from the boundary with the
next block) gives the right label. It fails silently whenever a state's
visitation count is under 50 — e.g. for `N_o_1 = [70, 5, 100]`
(`sum_no = [70, 75, 175]`, blocks `[1..70]`, `[71..75]`, `[76..175]`):
`kmkm(75-50)=kmkm(25)` overshoots state 2's tiny 5-sample block entirely
and reads state 1's label instead. This trick only exists because the
original code discards `kmedoids`'s second output (medoid *values*,
`[kmkm,~]=kmedoids(...)`) and tries to reverse-engineer the assignment via
index arithmetic instead. Once the medoid values are kept, assigning any of
the `n²` **distinct** states is just "nearest medoid to `V_o_1[i]`" — exact,
robust, independent of `N_o_1[i]`, and provably identical to what the
MATLAB produces whenever `N_o_1[i] ≥ 50` (i.e. it doesn't change behavior
in the regime the original code was presumably validated in) while fixing
the low-visitation failure mode as a byproduct rather than patching a
constant. Per your instruction, the original `-50` approach is kept as an
explanatory **comment only** in the ported code — not executed — so the
historical reasoning stays visible without carrying its failure mode
forward.

**Shape bug:** `ag_states_median = zeros(2, n²)` hardcodes 2 rows regardless
of `inf_bits` (should be `2^inf_bits` rows to hold all clusters). MATLAB's
auto-grow-on-assignment semantics mean writing to row 3+ (whenever
`inf_bits ≠ 1`, e.g. the `inf_bits=2` default used everywhere else in this
config) silently grows the matrix rather than erroring — so it "works" by
accident of MATLAB's array semantics, but the initial size is misleading
and worth getting right explicitly in the JAX port (allocate
`2^inf_bits` rows from the start, not 2).

Output shape/semantics: row `k` of `ag_states_median` lists every grid cell in cluster `k`
(zero-padded), and `EoC_SAIC_3Agents.m`'s `s_aggregate` looks a cell up in
this table at runtime to get its 1-indexed cluster id — that lookup **is**
the "learned communication policy" from §0.2.

### 4.4 What actually happens, step by step — run at `noa=2` per §0.8/ESAIC Theorem 1

```
noa = 2                                                   # NOT the decentralized phase's N — see §0.8
run ns-episode centralized Q-learning (§4.1)              # -> qp_table, N_table_emerged (noa=2, sidesteps §4.2's noa>=4 bug entirely)
V_o_1, N_o_1 = sum_q_MultiAgent(qp_table, ..., noa=2)     # §4.2
ag_states_median = aggregate_states_SAIC(...)             # §4.3
# Single batch only (§9 item 9, resolved) — MATLAB's bn=5 and its
# batch-1-only consumption by EoC_SAIC_3Agents.m are both superseded by
# just running the one batch actually needed.

# Then, per ESAIC Theorem 1: reuse this SAME 2-agent-derived ag_states_median
# for decentralized training at whatever N is actually wanted:
... decentralized training loop, noa=N (§5 — EoC_SAIC_3Agents.m) ...
```

This is the whole point of ESAIC (§0.8): the centralized/value/clustering
phase's cost is independent of the decentralized phase's agent count. The
`noa=4` config that appears as the default in `benchmark_perfectcom_MultiAgent.m`
is not this pipeline's actual input config — it's a separate, larger run the
ESAIC paper used to empirically demonstrate that training at `noa=4` directly
gives (approximately) the same `ag_states_median` as training at `noa=2` and
reusing it, i.e. Theorem 1's claim. Phase 1 doesn't need to reproduce that
validation experiment — it only needs the `noa=2` production path.

`benchmark_perfectcom_MultiAgent.m` lines 512–583 also contain a second,
separate block that recomputes state aggregation using `lloyds()` (Lloyd's
algorithm — a different, distortion-based scalar quantizer, not k-medoids)
against only the *last* batch's `qp_table`/`N_table_emerged`, disconnected
from the actual `batch_ag_states_median` pipeline `EoC_SAIC_3Agents.m`
consumes. **Confirmed not relevant — not porting it.** §4.1–4.3 (the clean
`aggregate_states_SAIC` function path) is the canonical recipe.

---

## 5. Decentralized training driver — `EoC_SAIC_3Agents.m`

This is the step-8-onward loop from Algorithm 1, and the actual reference
for `jax_saic/train.py`.

### 5.1 Setup (once, before the episode loop)

- Loads `ag_states_median` (§4.4's output) via `load(...)` — replaced in the
  JAX port by whatever §4 produces, threaded in as a plain argument, not a
  file load.
- `NE_table`, `NE_table_emerged`: UCB visitation counters, shape
  `(noa, n², 2^((noa-1)·inf_bits), 5)`, initialized to `0.001` (not `0`
  — avoids `log(·)/0` in the UCB bonus, see §6). Note the last-dim size:
  `2^((noa-1)·inf_bits)` is the number of distinct *joint* messages agent
  `i` can receive from all `noa-1` other agents (each sending `inf_bits`
  bits) — this is the size of the qp_table's message axis too.
- `cs = ones(noa, noa-1, inf_bits)` — initial dummy communication state.
- `legit_initial_pos = (1:n²) with goal_set removed` — episodes only ever
  initialize agents on non-goal cells.

### 5.2 Per-episode initialization

1. `tau = 1/(1 + i·tau_k)` for `i ≤ 40000`, else `tau = 1/(1 + 40000·tau_k)`
   — computed every episode but **unused** by the policy branch actually
   selected (see §6, dead-code flag).
2. `ps_ind = randi(n²-1, noa, 1)` → `ps = legit_initial_pos(ps_ind)` — RNG.
3. `ag_ps = s_aggregate(ps, ...)` — deterministic lookup into `ag_states_median`.
4. `pa = randi(5, noa, 1)` — RNG, but immediately overwritten at the first
   iteration of the while-loop before being read; this draw is pure wasted
   entropy, consumed but never used. Flagging, not removing.
5. `ca` initialized per `scen` (only `scen=3` matters here): `ca = randi(2,
   noa, bits) - 1` — RNG, and **also immediately overwritten** at the very
   first while-loop iteration (`ca = de2bi(ag_ps-1, inf_bits)`, deterministic).
   Another wasted-entropy draw. Both of these consume RNG state that a
   faithful JAX port's PRNGKey-splitting needs to account for if trying to
   trace MATLAB's rand-call sequence one-for-one — though per the task's own
   validation criteria, bit-exact RNG matching isn't the bar; this is
   documented for completeness, not because it needs replicating.

### 5.3 Per-step loop body (`while 3==3 ... break`)

Order of operations each step:

1. **Encode:** `ca = de2bi(ag_ps - 1, inf_bits)` — deterministic, no RNG.
   (`bits > inf_bits` branch with cyclic `encode`/`decode` is dead for our
   config per §0.5.)
2. **Channel (noiseless path, `bits == inf_bits`):** each agent receives the
   other `noa-1` agents' current `ca` rows verbatim (`bsc_ch` not called).
   For `noa > 2`, the `noa-1` received messages get concatenated per-agent
   into `cs_new`, shape `(noa, (noa-1)·inf_bits)`, for table indexing.
3. **Position update** (`pupdate_customized_nbits_ma`, skipped on the very
   first step of an episode, `counter(i) ≠ 1`): SARSA-style update of
   `qp_table` using `(last_ps, last_cs, pa, temp_rew)` from the *previous*
   step — see §7. If the *previous* step's `ter ≥ 1`, the episode's `while`
   loop breaks here, **after** this update (deliberate — the comment in the
   MATLAB source explains table updates must happen even after reaching the
   terminal state before the loop exits).
4. **Position action selection** (`ppolicy_customized_nbits_UCB_bestrew_ma`)
   — see §6. Consumes RNG (see §8).
5. UCB visitation counters `NE_table` (and, post-`end_learn`, `NE_table_emerged`)
   incremented for the `(agent, ps, message, action)` just selected.
6. Action is force-set to `5` (stay) for any agent already at `goal_set`.
   MATLAB checks this with raw `if ps(j)==[goal_set]` (line 273) — same
   implicit-`all()` bug as §2's fix; port uses `any(ps(j) == goal_set)` per
   your instruction, so this works correctly for a real multi-cell `goal_set`.
7. `envir_gc` steps the environment: `[ps, ~, ter] = envir_gc(ps, pa, n, noa, goal_set)`
   — using the fixed membership-test `ter` accumulation from §2.
8. **Stuck detection:** if every agent's position is unchanged vs. last
   step, `stuck_counter` increments; at `stuck_counter == 10` the episode
   force-terminates with `temp_rew = 0` and a one-off "unstick" hack: for
   every agent, `qp_table(agent, last_ps, last_message, pa)` is overwritten
   with the **median** of that state's entire action-row in `qp_table`.
   This is a heuristic to stop the agent from repeating the same
   self-defeating action — flagging per your instruction, not "fixing" it.
9. **Reward shaping — confirmed intentional, not a bug (corrected from the
   earlier draft):** `temp_rew` this step depends on `ter`:
   - `ter == noa` (all agents simultaneously at goal): `temp_rew = best_rew ^ (|rew_winner| - 1) = best_rew^(noa-1)` since `|rew_winner|=noa` here.
   - `ter ≥ 1` (partial arrival): same formula, `best_rew ^ (|rew_winner| - 1)`.
   - `ter == 0`: `temp_rew` stays whatever it was (default `0` at episode start).
   - `rew_winner` itself is built from raw `if ps(ii)==goal_set` in MATLAB
     (lines 358/367) — same §2 bug, same fix applied: `any(ps(ii) == goal_set)`.

   The earlier draft of this document flagged `best_rew^(noa-1)` (rather
   than a flat `best_rew`) as a likely carried-over bug from the 2-agent
   formula. **You corrected this:** it's deliberate reward *shaping* for
   the Q-table update signal (`temp_rew`), distinct from the actual task
   reward. The rationale: as `noa` grows, independent decentralized agents
   stumbling into simultaneous goal-arrival purely by exploration becomes
   exponentially rare, and even when it happens, it tends to happen late in
   the episode — so the discount factor `gamma^(counter(i)-1)` (applied
   downstream in §5.4's `rew(i)`) would crush its value to near-nothing,
   giving the Q-table almost no signal to learn "converge together" from.
   Exponentiating `best_rew` by `noa-1` artificially inflates the value of
   that rare, late, fully-synchronized outcome enough for it to actually
   propagate through the Q-table updates and shape behavior toward
   coordinated arrival. This is a training-signal trick, not the "true"
   reward: the actual task reward (what `EoC_SAIC_3Agents.m` reports/plots
   as `rew(i)`, §5.4) stays a flat, non-exponentiated `best_rew` vs. `1` vs.
   `0` — matching eq. 15 of the ESAIC paper exactly (`r = C2` full arrival,
   `C1` partial, `0` otherwise, `C1 < C2`; here `C2=best_rew`, `C1=1`). So
   there are legitimately two different reward-like quantities in this
   code, serving two different purposes:
   - `temp_rew` (this section): an artificially shaped training signal fed
     into the Q-table updates (§7) to make rare, valuable coordination
     events actually move the Q-values.
   - `rew(i)` (§5.4): the real, flat, eq.-15-matching task reward, used only
     for reporting/plotting — this is what the validation script's
     reward-vs-episode curve should reproduce, not `temp_rew`.
   Port `best_rew^(|rew_winner|-1)` for `temp_rew` verbatim — it's the
   intended behavior, general over `noa`, not a 2-agent-only formula that
   happens to also run at higher `noa`. This also explains, retroactively,
   why §4.1's centralized-training reward is flat with no exponent: a
   single centralized policy explores the *entire* joint action space
   directly rather than each agent exploring independently, so it never
   faces the "rare accidental synchronization" problem this shaping exists
   to solve — the two phases legitimately use different reward formulas,
   not one correct and one buggy.
10. `counter(i)` increments (step count for this episode).

### 5.4 Episode summary reward (`rew(i)`, used for the learning curve)

```
if term_hist(i) == noa:   rew(i) = best_rew · gamma^(counter(i)-1)   # all agents arrived together
elif term_hist(i) >= 1:   rew(i) = 1 · gamma^(counter(i)-1)          # partial arrival
else:                     rew(i) = 0                                 # stuck-terminated, nobody arrived
```

Note this summary reward is **not** `temp_rew` from the last step — it's a
separately-computed, simpler discounted reward keyed only off whether *any*
vs. *all* agents reached the goal, independent of the `best_rew^(|rew_winner|-1)`
formula used for the actual Q-table updates in-loop. Two different reward
signals coexist: `temp_rew` (drives the Q-updates) and `rew(i)` (the metric
plotted / returned for analysis, and what the validation script's moving-
average plot should reproduce).

---

## 6. Position policy — `ppolicy_customized_nbits_UCB_bestrew_ma.m`

Four policy modes exist in the code — `"ucb"`, `"greedy"`, `"ep_greedy"`,
`"q_prob"` (Boltzmann) — selected by a `policy` string argument. In MATLAB,
`EoC_SAIC_3Agents.m` line 240 hardcodes `policy = "ep_greedy"` immediately
before calling this function, unconditionally overwriting whatever
`policy` was passed into `EoC_SAIC_3Agents` itself — so no MATLAB caller
can actually select `"ucb"`/`"greedy"`/`"q_prob"` for the decentralized
loop no matter what they pass.

**RESOLVED (§9 item 7): this becomes a real, user-facing config parameter
in the port** — `decentralized_policy`, independent from the centralized
phase's own policy choice (§4.1 below), since the two functions have
different, non-overlapping option sets. The hardcode-override is dropped
(a deliberate, documented deviation from MATLAB, not a silent one): the
JAX port's `train.py` takes `decentralized_policy ∈ {"ep_greedy", "ucb",
"greedy", "q_prob"}` and actually respects it. `ep_greedy` remains the
default (matching what every existing MATLAB run effectively used), but
`ucb`, `greedy`, and `q_prob` are real, selectable, working options, not
present-but-unreachable dead branches.

### 6.1 `ep_greedy` branch (the default `decentralized_policy`)

```
epsilon = counter · (-0.98 / (ns · end_learn)) + 1
for each agent i:
    rr = rand()
    if rr <= epsilon:  pa(i) = randi(5)                              # explore
    else:              pa(i) = argmax_a qp_table(i, ps(i), msg(i), a) # exploit
```
where `counter` is the **episode index** `i` (confusingly reused name — the
outer episode loop variable in `EoC_SAIC_3Agents.m`, passed in as the last
positional argument), and `msg(i)` is the 1-indexed decimal value of the bits
agent `i` received (`cs_new` for `noa>2`, `cs(i,1,:)` for `noa==2`).

**This is the actual epsilon-annealing schedule** for Phase 1: linear decay
from `epsilon=1` at episode 0 to `epsilon=0.02` at episode `ns·end_learn`,
continuing to decay (unclamped) past that point — becomes negative for
`i > ns·end_learn/0.98`-ish, at which point the branch is unreachable
(`rr <= epsilon` never true for `rr∈[0,1]`, `epsilon<0`) i.e. functionally
floors at pure-greedy without an explicit `max(epsilon, 0)` clamp.

`tau` (the Boltzmann temperature computed every episode in
`EoC_SAIC_3Agents.m`, §5.2 step 1) is **only read by the `q_prob` branch**.
In MATLAB this made `tau`'s computation dead code (since `q_prob` was
unreachable); now that `decentralized_policy` is real and selectable,
`tau`'s annealing schedule is live and meaningful whenever
`decentralized_policy="q_prob"` is chosen, and inert (computed but unused)
under the other three options — this is normal, not a bug, since only one
policy is active per run.

### 6.2 UCB bonus (live whenever `decentralized_policy="ucb"` is selected)

```
ucb_const = 1.25 · best_rew / 3
bonus(a) = ucb_const · sqrt( log(ucb_counter+1) / NE_table(i, ps(i), msg(i), a) )
pa(i) = argmax_a [ qp_table(i, ps(i), msg(i), a) + bonus(a) ]
```
`ucb_counter = sum(counter(1:i))` — cumulative step count across all
episodes so far (computed once per while-loop iteration in
`EoC_SAIC_3Agents.m`, recomputed via a full `sum` over the whole `counter`
array every step — O(episodes) work every step, a performance smell but not
a correctness issue; not in scope to optimize per the task's non-goals).

RNG consumed by this function: `ep_greedy` draws one `rand()` per agent
(for `rr`), plus one more `randi(5)` per agent *conditionally* (only if
that agent's `rr ≤ epsilon`). Order: agent-major, `for i = 1:noa`, `rr`
then conditional `randi` before moving to the next agent. `ucb`/`greedy`
are fully deterministic given the Q-table/visitation counts and consume no
RNG at all; `q_prob` consumes one `rand()` per agent for its cumulative-
probability action draw.

---

## 7. Position update — `pupdate_customized_nbits_ma.m`

SARSA-style (on-policy, bootstraps off the action actually about to be
taken), tabular, `alpha = 0.07`, `gamma = 0.9` (both hardcoded inside the
function, **not** threaded from the caller's `gamma` argument — the caller's
`gamma=0.9` matches by coincidence of default value, but if a future config
ever changed the outer `gamma`, this inner hardcoded `0.9` would silently
diverge from it. Flagging.).

Three cases on `temp_rew` (computed in the caller per §5.3 step 9, passed in):

```
if temp_rew == 0:
    qp_table[i, last_ps, last_msg, pa] +=
        alpha · ( 0 + gamma · max_a qp_table[i, ps, msg, a] - qp_table[i, last_ps, last_msg, pa] )
        # standard bootstrapped TD update, no reward yet

elif temp_rew != 0 and len(rew_winner) == noa:   # only fires when ALL agents just won together
    for agent i in rew_winner:  qp_table[i, last_ps, last_msg, pa] += alpha · (temp_rew - qp_table[...])
    # note: the `else` branch inside this case (agent not in rew_winner) is unreachable
    # when the outer condition already requires len(rew_winner)==noa — dead code, harmless

else:  # temp_rew != 0 and NOT all agents won together (partial arrival)
    for all agents i:  qp_table[i, last_ps, last_msg, pa] += alpha · (temp_rew - qp_table[...])
    # RESOLVED: every agent's Q-value is nudged toward temp_rew here,
    # including agents that did NOT reach the goal this step. Confirmed
    # intentional, not a bug: the reward function is a single TEAM reward,
    # shared identically across the whole MAS (matches ESAIC eq. 15's
    # r(o_1,...,o_n,m_1,...,m_n) — one scalar reward for the team, not
    # per-agent individualized rewards). With a shared team reward there
    # is no separate credit-assignment scheme to apply: every agent's
    # Q-function legitimately incorporates the same team-level signal.
```

`last_msg` / `msg` here means: for `noa > 2`, `bi2de(cs_new_or_last_cs_new(i,:)) + 1`
(1-indexed decimal of the concatenated received-bits vector); for `noa == 2`,
`bi2de(transpose(squeeze(last_cs(i,1,:)))) + 1`. No RNG consumed anywhere in
this function.

`scen ∈ {1,2}` branches call a function `pupdate_customized_nbit` (singular,
no trailing `s`) that **does not exist anywhere in this repo** — dead/broken
code, but irrelevant since Phase 1 only ever runs `scen = 3`.

---

## 8. Hyperparameters and MATLAB defaults

From `parallel_simulator_2phase_encoded_MA.m` (only concrete-value caller of
`EoC_SAIC_3Agents` in the repo):

| Param | Value | Meaning |
|---|---|---|
| `scen` | `3` | Communication scenario (3 = the nbits/aggregated scheme; 1/2 not in scope) |
| `n` | `3` | Grid is `n × n` |
| `noa` | `4` in this file (comments give `2`→`ns=10000`, `3`→`ns=300000` for the same `n=3`) | Agent count — kept general, see §0.1 |
| `bits` | `2` | Total bits per channel use |
| `inf_bits` | `2` | Bits actually carrying information (`bits==inf_bits` ⇒ noiseless path, §0.5) |
| `goal_set` | `[9]` (`= n²`) | Single rendezvous cell |
| `best_rew` | `10` | Reward ceiling |
| `ns` | `1500000` (`noa=4`); `10000` (`noa=2`); `300000` (`noa=3`) | Episodes per run |
| `update_tables` | `1` | Boolean; gates all Q-table writes |
| `gamma` | `0.9` | Discount factor |
| `tau_k` | `0.005` | Boltzmann-temperature decay constant (dead in practice, §6.1) |
| `end_learn` | `0.80` (`end_learn_learn`) | Fraction of `ns` after which epsilon ≈ floor and `NE_table_emerged` starts accumulating |
| `bsc_p` | `1e-10` (`bsc_p_learn`/`bsc_p_exec`) | Effectively zero even in the runs that exist; consistent with §0.5's noiseless decision |
| `en` | `1` | Number of repeated epochs (outer loop over independent runs) |
| `policy` | `"ep-greedy"` (hyphen) at the call site | MATLAB's own string mismatch against `"ep_greedy"` (underscore, checked inside `ppolicy_...`) never mattered because of the hardcode-override bug (§6/§9 item 7). Both are moot now: `decentralized_policy` defaults to `"ep_greedy"` (correct spelling) as a real, working config parameter, §6. |

**`worst_rew` dropped entirely** (§9 item 8, resolved): confirmed via direct
grep that it appears only in `EoC_SAIC_3Agents.m`'s function signature and
is never read anywhere in its body, nor in any function it calls
(`ppolicy_customized_nbits_UCB_bestrew_ma.m`, `pupdate_customized_nbits_ma.m`,
or the centralized-training files). Not carried into the port at all, not
even as an accepted-but-ignored field.

### 8.1 Centralized-training defaults

From `Fully Centralized - MultiAgent/benchmark_perfectcom_MultiAgent.m` (the
only concrete-value caller of this phase):

| Param | Value | Meaning |
|---|---|---|
| `n` | `3` | Same grid as the decentralized phase |
| `noa` | `4` (comments give `ns` for `2`/`3` too — same pattern as §8's table) | See §4.2's `noa≤3` blocker, though |
| `ns` | `2000000` (`noa=4`); comments give `500000`(`noa=3,n=3`), `120000`(`noa=2,n=3`), `790000`(`noa=3,n=4`) | Episodes per batch |
| `bn` | `5` in MATLAB; **`1` in the port** (§9 item 9, resolved) | MATLAB ran 5 independent batches but `EoC_SAIC_3Agents.m` only ever consumed batch 1's result anyway (§4.4) — Phase 1 just runs the one batch needed |
| `end_learn` | `0.850` | Different value than the decentralized phase's `0.80` — two independently-set constants, don't conflate |
| `inf_bits` | `2` | Matches decentralized phase |
| `goal_set` | `9` (`=n²`) | Matches decentralized phase |
| `best_rew` | `10` | Matches decentralized phase |
| `gamma` | `0.9` | Matches decentralized phase (though hardcoded separately inside `pbench_update.m`, §4.1) |
| `tau_k` | `0.005` | Present in the script but **not used anywhere in this file at all** (not even computed) — fully vestigial here, more so than its already-mostly-dead counterpart in §6.1 |

---

## 9. Open items requiring your input before implementation

§0.7's discovery resolved the two biggest former unknowns (the empirical
`p(o₋ᵢ)` estimator and the clustering solver both now have real MATLAB
source, §4.2/§4.3); §0.8's discovery (ESAIC) resolved the `noa≥4` blocker
below. Sharper, more concrete open items remain in their place:

1. **RESOLVED by §0.8.** `decod_N_table` (inside `sum_q_MultiAgent.m`) only
   handling `noa ∈ {2,3}` looked like it blocked §4 at the decentralized
   phase's real agent count (`noa=4` default). It doesn't: per ESAIC
   Theorem 1, §4's centralized training/value/clustering always runs at
   `noa=2`, independent of the decentralized phase's `N` (§4.4). `noa=2` is
   fully supported by the existing code as-is — nothing to generalize or
   restrict.
2. **RESOLVED.** Clustering method is a runtime option, not a single fixed
   choice: `"kmedoids"` (MATLAB-fidelity heuristic) and `"kmedian"` (exact
   solve of eq. 13, default) both implemented and selectable — see §4.3.
   `kmedoids` was only ever MATLAB's available built-in, not the actual
   problem the paper states; no need to choose one over the other when both
   can just be options.
3. **RESOLVED.** The `-50` magic-number offset in `aggregate_states_SAIC.m`
   is not ported — replaced with direct nearest-medoid assignment for each
   of the `n²` distinct states (§4.3), which is exact, has no low-
   visitation failure mode, and is provably identical to the MATLAB output
   whenever the original `-50` trick would have worked correctly anyway.
   The original indexing approach is preserved as a comment in the ported
   code for historical traceability, not executed.
4. **RESOLVED.** The `lloyds()`-based block at the tail of
   `benchmark_perfectcom_MultiAgent.m` is not relevant — excluded from the
   port. §4.1–4.3's `aggregate_states_SAIC` path is the only clustering
   recipe in scope (with the `"kmedoids"`/`"kmedian"` option from item 2).
5. **RESOLVED — not a bug.** §5.3.9's `best_rew^(|rew_winner|-1)` exponent
   is confirmed-intentional reward *shaping* for the `temp_rew` training
   signal, general over `noa` by design (it counteracts the exponentially
   rare, late, discount-crushed nature of accidental simultaneous
   convergence as `noa` grows). Porting verbatim. The earlier draft's
   "corroborating evidence" from §4.1 (centralized training's flat reward)
   was a mistaken inference — the two phases use different reward formulas
   for a legitimate reason (§4.1/§5.3.9), not because one is buggy.
6. **RESOLVED — not a bug.** §7's partial-arrival branch updating *every*
   agent's Q-value toward `temp_rew` is confirmed-intentional: the reward
   is a single shared team reward across the whole MAS (matches ESAIC
   eq. 15), so there's no per-agent credit assignment to apply in the first
   place — every agent legitimately sees the same signal. Porting verbatim.
7. **RESOLVED.** `policy` becomes two independent, real, user-selectable
   config parameters — `decentralized_policy` (§6: `ep_greedy` default,
   plus `ucb`/`greedy`/`q_prob`) and `centralized_policy` (§4.1: `ep-greedy`
   default, plus `ucb`/`stochastic-ucb`/`stochastic-epsilon`/`stochastic`).
   All hardcode-overrides dropped as a deliberate, documented deviation from
   MATLAB. The centralized phase's `"stochastic"` option, which hardcoded
   broken 2-agent-only indexing in MATLAB, is fixed to generalize like its
   sibling branches (also per your instruction).
8. **RESOLVED.** `worst_rew` confirmed genuinely unused (grep-verified
   across `EoC_SAIC_3Agents.m` and everything it calls) — dropped entirely
   from the port, not kept as a vestigial field. See §8.
9. **RESOLVED.** Single batch, not `bn=5` — confirmed with the repo owner
   that one batch of centralized training is sufficient to produce the
   aggregated states. Matches what `EoC_SAIC_3Agents.m` actually consumed
   from MATLAB's 5 anyway (batch 1 only), just without running the other 4.
   See §4.4.

Nothing above blocks writing `env.py` / `channel.py` (§2/§3 are unambiguous),
and §4 is now unblocked at any target `N` (item 1 resolved via §0.8 — §4
always runs at `noa=2`). **All nine items in §9 are now resolved** — no
open judgment calls remain blocking Step 2.

---

## 10. Found only while implementing (Step 2)

Everything below surfaced during the actual JAX port (`jax_saic/`), not
during Step 1's read-only analysis. All ported faithfully (not silently
"fixed"), flagged here per the same discipline as everything else in this
document.

1. **CORRECTED, after further discussion with the repo owner.** The
   original entry here claimed the centralized phase's action-cancellation
   (`if sum(ps==goal_set)>=1: main_pa=5^noa`, `benchmark_perfectcom_MultiAgent.m`
   line 235-236) freezes the *whole team* the instant any one agent reaches
   goal, unlike the decentralized phase's per-agent freeze (§5.3 step 6) —
   framed as a real behavioral difference between the two phases.

   **Confirmed semantics (repo owner):** any agent entering `goal_set` marks
   that round as terminal; every agent still takes its action for that
   round (nothing aborts mid-round), and *then* the episode ends. This
   holds for both phases.

   Checking this against the actual loop structure (both scripts break as
   soon as `ter>=1`, not only at `ter==noa`, and initial positions always
   exclude goal cells) shows the "freeze" code in *both* scripts is
   **unreachable at runtime** — by the time a round could find an agent
   already sitting at a previously-reached goal cell, the episode has
   already ended. Verified empirically, not just re-argued: instrumented
   the ported centralized-policy path across 500 episodes (16,989
   action-selection checkpoints) and the freeze condition never fired once.
   So there is no actual behavioral difference between the two phases here
   — both simply apply the round's action to every agent, then check once
   for termination. The code is ported faithfully either way (matching
   MATLAB's own dead branch has no downside), but the earlier framing of
   this as a meaningful difference was wrong.
2. **Two separate Q-update call sites in the centralized loop, with an
   asymmetric learning-window gate — discussed with the repo owner,
   confirmed as an oversight, not a bug that changes qualitative behavior.**
   `end_learn` is a train/eval split over the *episode index* `i` (which
   episode, out of the total `ns`, we're on — there is no per-episode
   step-count cap anywhere in either script): the first `end_learn·ns`
   episodes are active learning, the remaining `(1-end_learn)·ns` let the
   learned policy run long enough to characterize its performance. Given
   that intent, both Q-update call sites should logically be gated by the
   same `episode < end_learn·ns` check — but only the terminal one (line
   299, carrying the actual `best_rew`/`1` reward) is; the unconditional
   top-of-loop one (line 215, zero-reward TD-bootstrap for every
   intermediate step) keeps firing regardless of episode index. Confirmed:
   this is an oversight — the guard should have applied to both.

   **Confirmed benign, not just "no observable difference":** past a
   converged Q-table, the bootstrap update `Q(s,a) += α(γ·max Q(s',·) −
   Q(s,a))` with no new reward is close to a no-op at its own fixed point —
   so continuing it during the eval tail mostly just lets any remaining
   value information keep propagating and settle, without disturbing the
   reward-carrying anchor values (those stopped changing once the terminal
   update was skipped). If the table hadn't fully converged by the
   `end_learn` cutoff, this tail phase incidentally helps finish that
   convergence. Not the intended effect, but not harmful either — ported
   verbatim (both call sites, asymmetric gate included) in
   `centralized.train()`, not "fixed."
3. **`rew_winner` confirmed dead everywhere (repo owner: only ever used for
   debugging, serves no real purpose) — dropped entirely in both phases,
   not just the centralized one.**
   - Centralized script: computed but never read at all (`temp_rew` there
     depends only on `ter`). Zero effect. Excluded entirely from
     `centralized.py`.
   - Decentralized script: used in two places, neither of which actually
     depends on the list itself. (a) `pupdate_customized_nbits_ma.m`'s
     `length(rew_winner)==noa` only selects *which* of two branches runs,
     and both compute the identical per-agent update once `temp_rew != 0`
     (§7) — no behavioral difference, so `qlearning.update()` never takes
     `rew_winner` as an argument. (b) The reward-shaping exponent
     (`best_rew^(|rew_winner|-1)`, §5.3.9) only ever needs the *length*,
     which is always identically equal to `ter` — both count "agents
     currently at any goal cell" via the exact same membership test — so
     `|rew_winner|` carries no information `ter` doesn't already give
     directly.

   Net effect: `rew_winner` is never constructed as a data structure
   anywhere in `jax_saic/` — every place MATLAB threads it through, the
   port already uses `ter` directly instead (mathematically identical, no
   information lost). Same treatment as `worst_rew` (§8): confirmed dead,
   not carried forward even as inert plumbing.
4. **Verified against source** (not assumed): MATLAB's stuck-counter break
   (`EoC_SAIC_3Agents.m` line 319) happens *before* `counter(i)=counter(i)+1`
   (line 387) — a stuck-terminated episode's `counter(i)` is never
   incremented for its final step. Has no effect on `rew(i)` (the
   stuck/else branch reports `0` regardless of `counter`'s value), but
   `jax_saic/train.py` matches the exact bookkeeping anyway rather than
   relying on that being true.
5. **Observed numerical artifact in the value-of-observation computation**
   (§4.2, ported exactly as documented — this is a property of the
   original formula, not a porting bug): the goal cell itself tends to end
   up with an anomalously *low* `V_o`, because episodes terminate almost
   immediately upon any agent's arrival — the centralized training loop's
   `N_table` (visitation counts) barely ever accumulates entries for a
   joint state where the marginalized agent is *at* the goal, since that
   joint state is reached and immediately exited in the same step. Since
   `prob_table` (§4.2) weights by visitation count, the goal state's
   contribution to `V_o` ends up small regardless of the underlying Q-value
   there. Observed directly in testing: with a 3×3 grid, goal cell (state
   8) got `V_o ≈ 1e-7` against neighbors' `V_o` in the 0.2–3.9 range, and
   landed in the same cluster as several low-value, far-from-goal states
   rather than being distinguished on its own. This is inherited from
   MATLAB's exact `max`-based `prob_table` formula (§4.2) — not something
   to silently patch (the correct fix, if any, is a paper/algorithm-level
   question, not a porting judgment call) — flagged for your awareness.
6. **`bsc_ch.m` generalized rather than replicated as-is** (`channel.py`):
   MATLAB's version only ever writes `cs(i,1,j)` — hardcoded to the single
   "other agent" slot, with its own comment acknowledging this
   (`"if noa increases, this line should be revisited! #noa"`). Since this
   function isn't wired into any trained pipeline (§0.5) and the MATLAB
   author's own comment flags the limitation as unfinished rather than
   deliberate, the port implements independent bit-flips across the full
   `(noa, noa-1, bits)` tensor instead of replicating the acknowledged gap.

---

## 11. Validation results (Step 3 — `validate/compare_to_matlab.py`)

Reference data comes from actually running the original algorithm files
under GNU Octave (installed in an isolated conda env `saic-octave`, no
system-level changes), not from re-deriving expected behavior by hand.
See `validate/octave_run_centralized.m`, `validate/octave_run_decentralized.m`,
and `validate/octave_shims/` for the compatibility layer this required:
missing `kmedoids`/`de2bi`/`bi2de` (never implemented in Octave-Forge's
`statistics` package, checked versions 1.4.3–1.8.4; Communications-Toolbox
functions Octave doesn't ship), a genuine Octave `sum(X, [dims])` multi-
dimension bug (verified in isolation), this Octave build's lack of support
for MATLAB-style local functions at the end of a script file (verified in
isolation), and the `-50` crash from §4.3/§10.5 — triggered live, for real,
confirming that finding wasn't hypothetical.

**Medium-scale comparison** (10,000 episodes, 8 seeds each side, `noa` ∈
{2,3,4}), each side running its own independent centralized training +
clustering (i.e. the actual end-to-end pipeline, no shortcuts):

| noa | jax_saic steady-state | Octave reference |
|---|---|---|
| 2 | 7.08 ± 0.03 | 7.46 ± 0.03 |
| 3 | 3.83 ± 0.62 | 4.41 ± 0.22 |
| 4 | 0.88 ± 0.01 | 0.84 ± 0.06 |

`noa=2`/`noa=4` track closely; `noa=3` showed a larger gap, worth
investigating given only 3 seeds/3000 episodes at the smaller first-pass
scale had shown an even bigger gap (jax 1.02 vs octave 2.18) that this
larger run already substantially narrowed — pointing at sample-size
variance rather than a port defect, but not conclusively.

**Anchored-`ag_states` experiment** (repo owner's suggestion): the
independent-clustering comparison confounds two things that can each
independently cause divergence — differences in the *decentralized
training* itself, and differences in *which `ag_states`* each side
happened to cluster into (each side runs its own centralized training +
clustering with a different RNG seed, per ESAIC's own design). To isolate
the decentralized phase specifically, both sides were fed the **identical**
`ag_states` (the one Octave's centralized run produced,
`validate/octave_work/central_reference.mat`, converted from MATLAB's
1-indexed/0-pad convention to this port's 0-indexed/-1-pad convention) —
see `validate/compare_anchored_ag_states.py`. Result:

| noa | jax_saic (anchored) | Octave reference |
|---|---|---|
| 2 | 7.47 ± 0.02 | 7.46 ± 0.03 |
| 3 | 4.29 ± 0.36 | 4.41 ± 0.22 |
| 4 | 0.87 ± 0.05 | 0.84 ± 0.06 |

All three agent counts now agree closely, and `noa=3`'s reward-vs-episode
curves overlap almost throughout training (not just in the tail) rather
than diverging late as in the independent-clustering run. **Conclusion:**
`qlearning.py`/`train.py` (the decentralized training logic) reproduce the
reference algorithm's behavior correctly; the residual gap in the
independent-clustering comparison was coming from each side's centralized
training + clustering converging to a different `ag_states`, not from a
defect in the decentralized port. This doesn't mean the two sides'
clustering are non-equivalent in some *other* respect worth chasing
further — just that decentralized training, given the same communication
policy, matches.

### 11.1 Centralized-training length sweep, and the `-50` bug's real-world impact

Requested directly: how long does centralized training (always `noa=2`,
SS0.8) need to run for the resulting `ag_states` to be trustworthy, and
does the original `-50` bug (§4.3, §10.5) actually matter in practice? Ran
`validate/sweep_central_length.py`: 4 values of `ns_central` (1000, 5000,
20000, 80000), each clustered two ways — the resolved nearest-medoid fix
(`"kmedian"`) and a faithful re-implementation of the *original* MATLAB
`-50` indexing bug (`method="legacy_minus50"`, `jax_saic/clustering.py`,
added for this ablation) — then each `ag_states` fed into decentralized
training (`noa=3`, 10,000 episodes, 4 seeds).

**The goal cell's visitation count (`N_o[8]`) is pinned at exactly 0.225 —
the initial smoothing floor — at every single `ns_central` tested (1000
through 80000).** Not shrinking, not growing. This confirms what §10.5
argued from static analysis, now with hard numbers across two orders of
magnitude of training length: the goal state's under-sampling isn't a
"hasn't trained long enough yet" problem — it's structural. Episodes end
the instant any agent arrives at goal, so "agent-1 is currently at the
goal cell, still selecting an action" essentially never happens as a
*current* joint state to accumulate visits into. No amount of additional
centralized training fixes this for the goal state specifically.

**Consequently, `ag_states` is *never* identical between the two methods
at any `ns_central` tested** — the legacy method misassigns the goal
state's cluster every time (confirmed: `misassigned_states=[8]` at 5000,
20000, and 80000). At `ns_central=1000`, the damage is much broader (8 of
9 states still under 50 visits, 2 literal crashes averted only by this
script's fallback, 3 more misassigned) — this is the "not trained long
enough" regime, and it's messy for reasons well beyond just the goal
state.

**Does the difference actually matter downstream?** Only at the very
short end:

| ns_central | understaffed (of 9) | kmedian fix | legacy `-50` |
|---|---|---|---|
| 1,000 | 8 | 4.00 ± 0.43 | 6.28 ± 0.54 |
| 5,000 | 1 (goal only) | 4.16 ± 0.25 | 4.15 ± 0.62 |
| 20,000 | 1 (goal only) | 4.13 ± 0.39 | 4.02 ± 0.42 |
| 80,000 | 1 (goal only) | 4.70 ± 0.29 | 4.29 ± 0.47 |
| 400,000 | 1 (goal only) | 4.89 ± 0.54 | 5.07 ± 0.58 |

A fifth point (`ns_central=400,000`, `validate/sweep_add_400k.py`, run
separately and merged into the same plot/table) reinforces both findings
rather than changing them. `N_o[goal]` is *still exactly* 0.225 at 400,000
episodes — bit-for-bit identical to its value at 1,000 episodes, a 400x
range with zero movement — about as strong a confirmation of "structural,
not undertrained" as this experiment could give. And at this point legacy
`-50` (5.07 ± 0.58) actually edges *above* the fix (4.89 ± 0.54), but well
within each other's std — read as more evidence the two are statistically
indistinguishable once only the goal state differs, with the direction of
the (noise-sized) gap flipping between points, not a consistent edge for
either method.

At `ns_central=1000` the two methods diverge sharply, but that's the
regime where 8 of 9 states are unreliable for either method — not a clean
read on the `-50` bug specifically, just evidence that undertrained
clustering is noisy however it's labeled (only 4 seeds here too, so this
particular number shouldn't be over-interpreted). From 5,000 episodes
onward, once the *only* remaining difference is the single goal-state
misassignment, **the two methods are statistically indistinguishable** —
every pairwise gap is well within one seed-to-seed standard deviation.
This makes sense given §10.5's characterization of that cluster slot: the
"agent already at goal" communication scenario is itself rare during
decentralized training for the same structural reason it's rare during
centralized training (episodes end on arrival), so which cluster it lands
in barely gets exercised.

**How long is "long enough"?** Using the fix's own numbers as the cleanest
read (unconfounded by the `-50` bug): 4.00 → 4.16 → 4.13 → 4.70 → 4.89
across 1,000 → 5,000 → 20,000 → 80,000 → 400,000. Most of the benefit is
captured by 5,000 episodes (matching the point where only the
structurally-unfixable goal state remains understaffed); 20,000 doesn't
clearly beat 5,000 (within noise); from 80,000 to 400,000 (a 5x increase)
the gain is small (4.70 → 4.89) and comparable in size to the seed-to-seed
noise at these scales — consistent with diminishing returns setting in,
not a sharp plateau. Practical recommendation: **5,000–20,000
centralized-training episodes is a reasonable default** for this `n=3`
config — enough that every state except the structurally unfixable goal
cell is well-estimated, capturing most of the achievable benefit — with a
further, small, still-real-looking gain available from training an order
of magnitude longer if the extra compute is cheap (centralized training is
`noa=2`-only and fast regardless, so it usually is), but not worth it if
compute is the binding constraint.

**Toward a methodical replacement, not just a bigger magic number:** the
`-50` trick was implicitly trying to answer "which medoid does this
state's value belong to," but via an indirect route — replicate-by-count,
run PAM, then read a label back out through an index computed from
cumulative sample counts — that only works when there happen to be enough
replicated copies to guarantee the index lands inside the right block.
That indirection is *why* it needs a visitation-count threshold at all,
and why any fixed constant is fragile: it's a proxy for "enough samples to
trust the block boundary," and proxies break exactly on states like the
goal cell whose rarity is structural, not a sample-size accident no
constant can fix. The resolved fix (`nearest_medoid(V_o[i])`, direct)
doesn't have this problem because it never goes through the replicated-
array indirection at all — it depends only on the state's own value and
the (already jointly-determined) medoid positions, so it's well-defined
at *any* visitation count, including zero. That's the methodical
replacement: not a better threshold, but removing the step that needed
one.

(Aside: one raw log line from this run shows a single decentralized seed
"done in 12109.3s" (~3.4h) against ~85s for every sibling run at the same
settings — this is a wall-clock-timing artifact from the laptop being
suspended (lid closed) mid-run, not a real anomaly. Suspend-to-RAM freezes
the process without corrupting its in-memory state, so the underlying
computed reward curve for that seed is still valid; only the printed
wall-clock duration is inflated by the sleep time.)

### 11.2 Normalizing decentralized return against the exact optimal ceiling

Requested directly: put the measured decentralized steady-state returns
into perspective by normalizing against the best achievable return —
computable in closed form, no RL needed, since the environment is fully
deterministic (`validate/optimal_return.py`).

**Derivation (an earlier version of this was wrong, caught before
reporting it — recorded here since the mistake is instructive):** the
first pass assumed an agent could only delay its arrival in *even*
increments (a "detour there and back"), reasoning that the grid graph is
bipartite so path length parity is fixed by the start-goal pair. That's
true *of paths*, but it ignored the **STAY** action entirely — since STAY
leaves position unchanged, an agent can delay by exactly *one* step at a
time with no parity restriction at all, as long as the padding happens
before it ever touches the goal cell (touching goal ends the episode
immediately, so an agent can't safely "visit and leave" to pad *after*
arriving — confirmed the hard way, by first writing a verification
simulator that tried exactly that and caught the bug in dry-testing).
**This was caught by the numbers, not by re-reading the derivation**: the
first-pass formula produced a `noa=2` "optimal" of 4.21 — lower than the
already-measured trained return of ~7.08–7.47, which is impossible for a
true upper bound. That contradiction is what triggered a re-derivation
rather than reporting a broken ceiling.

**Corrected result:** since STAY-based padding has no parity restriction,
every agent can always pad to arrive at exactly `T = max_i(d_i)` (its
shortest-path distance to goal), so full-team simultaneous arrival is
*always* achievable — optimal reward is unconditionally `best_rew ·
γ^max_i(d_i)`, averaged over all `(n²-1)^noa` equally-likely starting
tuples (enumerated exactly — 512 for `noa=3`, `n=3` — not sampled).
Verified against a scripted (non-learned) simulator that actually walks
the environment with STAY-padded shortest paths: 0 mismatches across 300
random trials at each of `noa=2,3,4`, and separately cross-checked the
`counter → T` bookkeeping this depends on by mirroring `train.py`'s exact
loop logic on a hand-built trajectory (`ep_rew=9.0` for a `T=1` full
arrival with `best_rew=10, gamma=0.9`, matching `best_rew·γ^1` exactly).

| noa | optimal (exact) | jax_saic (measured) | Octave reference (measured) |
|---|---|---|---|
| 2 | 7.492 | 7.082 ± 0.031 (94.5%) | 7.455 ± 0.029 (99.5%) |
| 3 | 7.261 | 3.834 ± 0.617 (52.8%) | 4.407 ± 0.221 (60.7%) |
| 4 | 7.115 | 0.880 ± 0.010 (12.4%) | 0.838 ± 0.057 (11.8%) |

**The finding this surfaces:** the optimal ceiling itself is nearly flat
across `noa` (7.49 → 7.26 → 7.12, only ~5% down from `noa=2` to `noa=4`)
— `max_i(d_i)` over more i.i.d.-sampled agents grows slowly. The steep
drop in raw measured return with `noa` is therefore **not** the
rendezvous task itself getting harder — it's the decentralized,
bit-budgeted coordination problem getting dramatically harder relative to
an almost-unchanged ceiling. Efficiency (measured/optimal) goes from
~95–99% at `noa=2` to ~12% at `noa=4`. This reframes every `noa`-sweep
plot in this document: the right takeaway from the earlier `reward_vs_noa`
plots isn't "returns fall because the task gets harder," it's "decentralized
coordination efficiency collapses much faster than the task itself would
require."

### 11.3 Is the centralized-training foundation itself any good?

Asked directly, in response to §11.2's efficiency numbers looking poor at
`noa=3`/`4`: before blaming decentralized coordination, is the centralized
(`noa=2`) phase itself actually converging well? If it weren't, the VoI
signal feeding clustering would be low-quality, and *that* could explain
part of the shortfall independent of decentralized-specific difficulty.

`centralized.py`'s `train()` previously returned only `(qp_table,
N_table_emerged)` — it never tracked the reference script's own `rew(i)`
metric at all, since the pipeline only needed the Q-table for value-of-
observation. Added reward-tracking (now returns `(qp_table,
N_table_emerged, rew)`, matching `benchmark_perfectcom_MultiAgent.m`'s own
`if temp_rew>1: rew=best_rew*gamma^(counter-1); else: rew=1*gamma^(counter-1)`
— no stuck-escape branch exists in the centralized script, confirmed via
source grep, so this is never `0`) to answer the question empirically
rather than assuming.

**Result, `ns=80,000`:** steady-state return `7.464 ± 0.80`, against an
exact optimal of `7.492` — **99.6% of optimal**, converging cleanly
(saturating by ~65,000–70,000 episodes, `validate/check_central_return.py`,
`plots/central_reward_vs_episode.png`).

**Conclusion:** the centralized-training foundation is not the source of
the efficiency drop seen in §11.2. It's about as good as it can be. The
degradation at `noa=3` (~53–61%) and `noa=4` (~12%) is specifically the
decentralized, bit-budgeted coordination problem getting harder as team
size grows — not inherited weakness in the value-of-observation signal
the clustering step depends on.

### 11.4 Does value propagation actually stall once epsilon anneals?

Sharper version of §11.3's question, raised directly: achieving
near-optimal *reward* doesn't imply every state's *value* is precisely
estimated. Q-learning only refines `Q(s,a)` for `(s,a)` pairs actually
visited, and once epsilon anneals toward 0, an agent mostly re-treads
whatever trajectory the greedy policy already commits to — states off
that trajectory could in principle go stale. This matters specifically
for VoI-based clustering, which needs precise values for *every* state,
not just the ones on an optimal path.

Checked empirically rather than argued: since the environment is fully
deterministic and known (same fact behind §11.2's closed-form optimal
return), the *exact* value function can be computed directly via value
iteration on the known MDP (`validate/check_value_precision.py`,
`exact_value_iteration()`) — no Q-learning, no exploration, no sampling
noise at all — and compared against what `ns=80,000` Q-learning actually
produced. To isolate Q-value precision specifically (not a side effect of
a different weighting scheme), both were passed through
`value_of_observation()` using the *same* empirical `N_table_emerged`
weighting from the learned run.

**Result: exact and learned `V_o` match to 4 decimal places at every one
of the 9 states**, including the goal state (both exactly `0.0000`), and
the resulting `ag_states` clustering is bit-for-bit identical between the
two. (One "99.8% relative error" figure for the goal state in the raw log
is a numerical artifact of dividing by ~0, not a real discrepancy — both
values agree to 4 decimals.)

**Why the theoretical concern doesn't bite in practice here:** the
initial joint position is re-randomized every episode
(`jax.random.randint(...)`), even after epsilon has annealed to ~0 — so
every state keeps getting revisited via fresh starts, not through
in-episode exploration. Combined with a tiny state space (81 joint states
for `noa=2`, `n=3`) and 80,000 episodes of training, that's enough
continued coverage for full convergence well before training ends. The
concern is real in general (a larger state space or fewer episodes could
absolutely leave this gap open — worth re-checking if `n` or the episode
budget changes materially), but for the configuration actually used
throughout this document, it's verified closed, not assumed closed.

Combined with §11.3, this closes the "is the centralized foundation
trustworthy" question from two independent angles — matching the true
optimal *reward* (99.6%) and matching the true optimal *value function*
(exact, to 4 decimals) — rather than one. The `noa=3`/`noa=4`
decentralized shortfall is not inherited from the centralized/VoI/
clustering phase under any measure checked so far.

### 11.5 A real precision problem in V_o, found by inspecting the Fig-8-style clustering directly — and a fix, not just a diagnosis

§11.4 verified that Q-learning estimates `Q(s,a)` correctly *given* the
empirical visitation-count weighting `value_of_observation()` uses to
marginalize over the other agent's position. It did **not** verify that
this *weighting itself* is precise — and generating the Fig-8-style grid
visualization (`validate/plot_grid_clusters.py`) surfaced a concrete case
where it isn't: states 5 `(2,1)` and 7 `(1,2)` are mirror images across
the diagonal through the goal at `(2,2)` — both exactly 1 step away —
so a correct value-of-observation must treat them identically. Measured:
`V_o[5]=2.637` vs `V_o[7]=1.628`, a **38% relative gap**, far larger than
the other two symmetric pairs in this grid (`(1,3)`: 14%, `(2,6)`: 10%).

**Root cause, confirmed not just Q-value error:** `value_of_observation()`
weights the marginalization by `N_table_emerged` — a visitation count
sampled from a *single, limited-window* (post-`end_learn` only) training
run. That weighting is inherently seed-dependent and noisy, entirely
independent of whether the underlying Q-values are accurate (§11.4 already
showed those match exactly). Repo owner's framing, and it's the right one:
even if this is "just seed noise" that would shrink averaged across many
independent training runs, the actual pipeline only ever runs *one* seed
— "it would average out" isn't a fix, since we don't perform that
averaging in practice.

**The fix — remove both noise sources instead of reducing them:**
both agents' starting positions are drawn *independently* and *uniformly*
over the same legal (non-goal) cells. That's the actual, exactly-known
distribution the marginalization is supposed to approximate — it doesn't
need to be estimated from a sampled training run at all.
`validate/exact_value_of_observation.py` computes value-of-observation
with zero variance by construction:
  - **Exact Q\*** via direct value iteration on the known, deterministic
    MDP (§11.4's `exact_value_iteration`, already verified) — no
    Q-learning, no exploration, no sampling.
  - **Exact uniform marginalization** — average `V*(joint state)` over
    the other agent's position with weight `1/(n²-1)` for every legal
    cell, in closed form, instead of empirically-sampled visitation
    counts.

Verified: all three symmetric pairs now match to floating-point precision
(`diff=0.00e+00`), not just "closer." And the values themselves changed in
a way that makes much more sense, not just less noise around the same
numbers: the goal cell's value-of-observation was `0.00` under the
empirical/sampled approach (§10.1/§10.5's structural under-sampling
artifact — the goal cell is almost never a "current" state to weight,
since episodes end the instant it's reached) and is now `8.81` — one of
the *highest* values, matching the obvious intuition that being at the
goal should be valuable. The resulting clustering (regenerated in
`validate/plot_grid_clusters_exact.png`) changed from a somewhat
arbitrary-looking grouping (goal lumped in with the lowest-value corner)
to clean, symmetric distance-rings radiating from the goal: `{0}`,
`{1,3}`, `{2,4,6}`, `{5,7,8}` — the goal cell correctly joins its nearest
neighbors as the highest-value tier.

**Scope of this fix:** it's specific to this environment's structure —
both agents' starts are i.i.d. uniform and independent by construction
(`train.py`'s episode init), so the exact marginal distribution is known
in closed form. That won't generalize automatically to an environment
where the "other agent's position given mine" genuinely has an unknown or
correlated distribution (e.g., a non-uniform start distribution, or later
timesteps where two agents' positions become correlated through a shared
policy) — there, some form of estimation would still be needed, just
hopefully a better-justified one than "whatever a single training run's
visitation counts happened to produce." For *this* phase's problem
(`noa=2` centralized MDP, i.i.d. uniform independent starts, evaluated at
episode start), the exact closed-form approach is strictly better and
should in principle replace the empirical one for this case.

**Decision (explicitly asked, 2026-07-30): kept out of production for now.**
`exact_value_of_observation.py` stays a standalone `validate/` diagnostic;
`jax_saic/clustering.py` and `train.py`'s `run_phase1()` are unchanged and
still use the empirical, visitation-weighted `value_of_observation()`. If
this is revisited later, the natural integration point is a new
`method="exact"` (or similar) branch in `cluster_states()`, gated to the
`noa=2`-centralized/i.i.d.-uniform-start case this derivation assumes.

**Follow-up, same day: `exact_value_of_observation.py` doesn't actually
generalize, and here's the general replacement.** Raised directly: SAIC/
ESAIC's whole point is to be a general multi-agent-coordination method,
not something hand-solved for rendezvous specifically -- so "assume the
other agent's position is uniformly distributed" can't be the real fix,
even though it happens to give a clean, symmetric answer on this
particular symmetric grid. It conflates two different things: the
*initial*-position distribution (which genuinely IS i.i.d. uniform over
non-goal cells, confirmed by reading `centralized.py:217-221` -- that's
just the algorithm's own init code, fair to use in any application, not
task-specific knowledge) with the *whole-trajectory occupancy*
distribution `N_table_emerged` is meant to approximate, which has no
general reason to stay uniform past t=0.

The general fix, in `validate/exact_occupancy_value_of_observation.py`:
don't assume any marginal shape at all. Forward-propagate the known
i.i.d.-uniform initial distribution through the known, deterministic
transition dynamics (`env.step` -- already required by value iteration
itself, so no new assumption) under the policy that's actually greedy
w.r.t. the already-exact Q*, and read off the resulting exact occupancy
measure. This requires nothing rendezvous-specific: any tabular, finite,
model-known SAIC/ESAIC application has (a) known dynamics, (b) a known
episode-init distribution (it's in the training script), and (c) a greedy
policy derived from whatever Q* it already computes.

First attempt at this (single deterministic rollout per start pair,
breaking ties by lowest raw joint-action index) made the asymmetry
*worse*, not better -- checked directly: 73 of the 81 joint states have
multiple tied-optimal actions (moving toward the goal x-first vs. y-first
is equally optimal almost everywhere), and the raw action-index ordering
(`RIGHT,LEFT=0,1` before `UP,DOWN=2,3`) has no reason to respect the
grid's diagonal symmetry, so an arbitrary single-path tie-break silently
picks one asymmetric optimal trajectory out of many equally-valid ones.
Fixed by treating ties properly: propagate probability *mass* through the
state space, splitting it uniformly across every tied-optimal action at
every step (a well-defined "greedy, uniform tie-break" policy, evaluated
exactly rather than sampled), instead of picking one path. Verified:
`residual_mass=0.00e+00` (every unit of probability mass reaches the goal
within the step cap) and all three symmetric pairs match to floating-point
precision again.

**Corroborating evidence, same day: the empirical asymmetry is NOT just
noise that averages out.** `check_symmetry_multiseed.py` (launched earlier,
5 independent seeds, `ns=80,000` each) finished: the across-seed MEAN for
the (5,7) pair is `V_o[5]=2.730` vs. `V_o[7]=1.689` -- still a 38% relative
gap, essentially unchanged from any single seed (individual seeds ranged
31%-69%). If this were ordinary sampling noise, averaging 5 independent
runs should have shrunk the gap noticeably; it didn't move at all. The
other two symmetric pairs, by contrast, DID average down to a small gap
(`(1,3)`: 2%, `(2,6)`: 3%) -- consistent with those being genuine noise
that (1,3)/(2,6) are less exposed to for whatever structural reason, while
(5,7) has a persistent, non-shrinking bias. This rules out "just run more
seeds" as a fix and is exactly the failure mode the general
occupancy-based method (above) is built to eliminate structurally rather
than average away.

The resulting numbers are meaningfully different from the assumed-uniform
version, not just less noisy — occupancy concentrates near the goal (e.g.
`N_o[8]=1.0` vs. `N_o[0]=0.125`, since every trajectory passes near the
goal at the end but few pass through the far corner), which pulls the
goal's `V_o` further above its neighbors (`V_o[8]=10.0` vs. `V_o[5]=V_o[7]
=9.66`, vs. the assumed-uniform version's `8.81`/`8.81`). Reclustering on
this gives `{0,1,3}, {2,4,6}, {5,7}, {8}` (`plot_grid_clusters_occupancy.png`)
— goal now correctly earns its own singleton cluster (max-4-cluster
regime, `inf_bits=2`) rather than being pooled with its near neighbors,
which the assumed-uniform version's clustering had done.

Same production-integration decision applies here as above (kept as a
`validate/` diagnostic, not wired into `clustering.py`, pending the same
kind of explicit confirmation) — but this occupancy-based version, not the
assumed-uniform one, is the one that should actually be considered if/when
that integration happens, since it's the only one of the two that doesn't
smuggle in a rendezvous-specific assumption.

### 11.6 Does decentralized training itself close the gap, given perfect aggregation? Yes — the bottleneck is clustering, not training logic.

`validate/decentralized_on_exact_agstates.py` (its own docstring, written
same session as §11.5) frames the exact question directly: "worth running
decentralized training phase on top of the obtained aggregation from this
diagnostic tool, to see if we get close to the optimal solutions or not —
if we don't, the problem is in the decentralized training phase, as the
aggregation is perfect here." This section reports that run's result
(2026-07-30/31, run in `validate/`, not wired into any pipeline module).

**Setup:** `ag_states` fixed to the exact, zero-variance clustering from
`exact_value_of_observation.py` (the *assumed-uniform* version from
§11.5's first fix — `{0}, {1,3}, {2,4,6}, {5,7,8}` — not the more general
occupancy-based one from §11.5's follow-up; this script predates that
follow-up and wasn't re-pointed at it). Decentralized training then run
at `noa=2` (`ns=10,000`, 5 seeds), `noa=3` (`ns=300,000`, 5 seeds) via
`decentralized_on_exact_agstates.py`, and `noa=4` (`ns=1,500,000`, 1 seed
only — by far the most expensive leg, so the sweep script's own noa=4 run
was killed right after noa=3 finished, via a background watcher polling
its log, to avoid duplicating that compute) via a new one-off script,
`validate/plot_noa4_reward_curve.py`, which also saves the raw per-episode
reward array (`validate/noa4_reward_curve.npy`) and a 10,000-episode
moving-average plot (`validate/plots/noa4_reward_vs_episode_exact_agstates.png`)
— `decentralized_on_exact_agstates.py` itself only prints summary stats
and discards the per-episode trajectory.

**Result**, efficiency = steady-state tail mean / `optimal_return.py`'s
closed-form optimal:

| noa | steady-state | optimal | efficiency | seeds |
|---|---|---|---|---|
| 2 | 7.486 | 7.492 | 99.9% | 5 |
| 3 | 7.247 | 7.261 | 99.8% | 5 |
| 4 | 7.067 | 7.115 | 99.3% | 1 |

All three at or above 99%, essentially flat across `noa` — compare to the
empirically-clustered baseline (`plot_normalized_return.py`,
§11.2/§11.3): ~94–99% at `noa=2`, ~53–61% at `noa=3`, ~12% at `noa=4`.

**Conclusion:** decentralized-training logic is not the bottleneck at
higher `noa` — this was already the working hypothesis behind §11.5's
investigation, and this run confirms it directly rather than just
plausibly. Given a perfect aggregation, `noa=4` decentralized training
reaches 99.3% of optimal; given the normal empirically-clustered
aggregation, it reaches ~12%. The entire gap is attributable to
value-of-observation/clustering imprecision (§11.5), not anything in
`qlearning.py`/`train.py`'s decentralized update logic. `noa=4` here used
only 1 seed (cost — 5 seeds at `ns=1,500,000` was judged not worth it for
this confirmatory check), so no seed-variance estimate the way `noa=2`/`3`
have, but a single-seed 99.3% is decisively far from the ~12%
empirical-clustering result — seed variance at this scale would not
change the qualitative conclusion.

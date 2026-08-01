# CERTIFICATION.md — boundary-gate value certification + refinement for ESAIC

Additive to the Phase-1 SAIC/ESAIC port (`jax_saic/`; see `CLAUDE.md`/
`PORT_NOTES.md`). Lives under `esaic/`, opt-in via
`CertificationConfig.cert_enabled` (default `False`) — nothing here changes
existing pipeline behavior unless explicitly turned on. See
`PORT_NOTES.md` §11.5/§11.6 and the memory note `centralized_precision_at_scale`
for the motivating problem: `jax_saic/clustering.py`'s `value_of_observation()`
is only accurate in expectation under the centralized policy's own
occupancy, not uniformly over the observation space Ω — but the k-median
clustering step needs accuracy specifically *near cluster boundaries*, for
every observation any decentralized policy might visit, or codewords get
mis-assigned.

## 1. The margin gate

For medoids `mu` (shape `[B]`) and a value estimate `V_hat` (shape `[|Ω|]`),
the margin of observation `o` is its distance to the midpoint of its two
nearest medoids:

```
m(o) = 0.5 * (d_runner_up(o) - d_nearest(o))
```

`o` keeps its correct codeword as long as `V*(o)` stays on the same side of
the nearest boundary as `V_hat(o)`; a **sufficient** condition is
`|V_hat(o) - V*(o)| < m(o)`. Value error only ever matters for the
assignment through this ordering (Singh & Yee 1994; Williams & Baird 1993)
— so certification gates on `m(o)`, not a global sup-norm.

A **certificate** `U(o)` is any nonnegative, model-free, sample-based
quantity satisfying `U(o) >= |V_hat(o) - V*(o)|` with high probability.
`m(o)` is inflated defensively by the largest medoid shift seen across
refinement rounds (`delta_medoid`, since `m(o)` is computed from
`V_hat`-based medoids, not `V*`-based ones, and those medoids move as
refinement retrains):

```
m_eff(o) = max(0, m(o) - margin_inflation * delta_medoid)
gate:     certified(o) = U(o) < m_eff(o)
```

`certified(o) == True` implies `o`'s codeword provably equals its
`V*`-codeword (w.h.p.). `F = {o : not certified(o)}` is what refinement
targets (`esaic/certification.py`: `margins`, `certify`).

## 2. Two certificate forms

### 2.1 Bracket (`cert_mode="bracket"`, default)

```
U_bracket(o) = max_a Q_ub(o,a) - V_pi_lower(o)
```

- `Q_ub(o,a)`: an optimistic upper Q, built by inflating the learned joint
  (`NOA=2`) Q-table with a Hoeffding/Bernstein-style UCB bonus
  (`ucb_scale * Rmax/(1-gamma) * sqrt(log(N_total)/N(s,a))`, discounted
  variant; Jin, Allen-Zhu, Bubeck & Jordan 2018) so `Q_ub >= Q*` w.h.p.,
  then marginalized down to a single agent's `(o, a)` slice.
- `V_pi_lower(o)`: an unbiased Monte-Carlo return of the **current greedy**
  joint policy, rolled out from a joint state with the agent under test at
  `o` and the partner sampled from the **same empirical conditional
  distribution** `V_hat`'s own marginalization uses (see §3 below — this
  match matters). `V_pi_lower(o) <= V*(o)` always, since a greedy policy
  w.r.t. a possibly-imperfect Q is suboptimal.

Avoids the Bellman residual entirely — its single-sample square is biased
by the double-sampling term (Baird 1995).

**Why the bracket is sound even when `V_hat` itself is untrustworthy**:
if `Q_ub(o) >= V*(o)` and `V_pi_lower(o) <= V*(o)` both hold, then `V*(o)`
lies in `[V_pi_lower(o), Q_ub(o)]`. If `V_hat(o)` also lies in that
interval, `|V_hat(o)-V*(o)|` is at most the interval's width, `U_bracket`.
Tested directly (`tests/test_certification.py::test_bracket_ordering`):
at modest training budgets, `V_hat` can fall *outside* that interval for
chronically under-visited-as-a-current-state observations (the rendezvous
task's goal cell — `PORT_NOTES.md` §11.5's own finding: episodes end the
instant it's reached, so it rarely gets Q-updates as a *starting* state).
When that happens, the interval-containment argument above doesn't apply
cleanly — but `Q_ub`'s bonus inflates exactly where visitation is low, and
empirically the primary bound (`U_bracket >= |V_hat-V*|`, checked against
exact value iteration as a test oracle) held for all 9 observations even
when the softer ordering held for none of them. The "sandwich" ordering is
a **sufficient, not necessary**, condition for soundness.

### 2.2 Concentration (`cert_mode="concentration"`)

```
U_conc(o) = z_(1-alpha) * sigma_hat(o) + b_max(o)
```

- `sigma_hat(o)`, two sources (`sigma_source`):
  - `"ensemble"` (default): `std` over `ensemble_size` independently-seeded
    centralized reruns' `V_hat(o)` — captures approximation/training
    variance, not just within-run sampling noise. Cost: `O(ensemble_size)`
    full centralized retrains, independent of `N`.
  - `"count"`: `c * Rmax / ((1-gamma) * sqrt(n(o)))`, grounded in tabular
    ℓ∞ theory (Wainwright 2019, arXiv:1905.06265; Li, Wei, Chi, Gu & Chen,
    IEEE Trans. IT 68(1):448–473, 2022; Qu & Wierman, COLT 2020) — entrywise
    error scales with per-entry visitation, governed by `mu_min` (the
    minimum occupancy). Cheap — no extra training, reuses `N_o`.
- `b_max(o)`: maximization-bias budget. `0` when `use_double_q=True`
  (double-Q removes the bias structurally); otherwise estimated as
  `max(0, V_singleQ(o) - V_doubleQ(o))` from a short paired run.

## 3. Why the partner-position distribution matters (a correction made during implementation)

`V_hat`'s marginalization over the partner agent's position uses an
empirical **conditional** distribution, `P(partner=j | self=o) =
N_state(j,o) / sum_j' N_state(j',o)`, derived from the joint visitation
counts — matching `PORT_NOTES.md` §0 item 6's deliberate choice
("estimated empirically from centralized-training rollout visitation
frequency, not assumed uniform").

An earlier version of this module reused `jax_saic.clustering.
value_of_observation()`'s exact weighting formula (each joint state's
"probability" approximated by its *max-visited-action* count divided by
the *total* visitation count across the whole joint table) — this is
explicitly flagged in `PORT_NOTES.md` §4.2/§11.5 as SAIC's own
approximation, and it does **not** sum to 1 over the partner's position
for a fixed `o`. Reusing it made `V_hat` systematically not equal to the
greedy policy's actual expected value, which broke `test_bracket_ordering`
outright (`V_pi_lower(o) > V_hat(o)` for 9/9 observations). Fixed to a
properly-normalized conditional distribution (`esaic/certification.py`:
`_marginalize_joint_q`), and — critically — `bracket_v_pi_lower`'s Monte
Carlo rollouts now sample the partner's starting position from that same
distribution, rather than from the environment's natural (typically
uniform) reset distribution (`esaic/certification.py`:
`_conditional_partner_distribution`, `_mc_rollout_return`). Comparing
`V_hat` and `V_pi_lower` under two different definitions of "the other
agent's position given mine" was the root cause, not a tolerance issue.

This remains fully generic: the conditional distribution is derived only
from visitation counts (`N_o`) and `n_obs`, with no task-specific content.

## 4. Refinement loop

```
1. centralized_train(...)          -> V_hat, N_o, (qp_table)
2. cluster_states(V_hat, N_o, ...) -> mu, ag_states / cluster_id
3. certify(...)                    -> certified, m_eff; F = uncertified set
4. while F nonempty, budget remains, rounds remain:
       retrain with agent-1 episode-starts biased toward F
       (focus_states=F, focus_frac=refine_oversample_frac)
       re-cluster; delta_medoid = max shift in sorted medoid values
       re-certify
5. apply uncertified_policy to any observations still in F:
     "flag" (default) -- leave as-is, return the certified mask
     "conservative"    -- reassign to the LOWER-value of its two nearest
                           medoids (fail toward under-, not over-estimate)
     "weighted"        -- re-cluster with 1/U(o) replacing N_o as the
                           k-median sample weight, so unresolved points
                           pull the medians less
```

`esaic/generic_centralized.py`'s trainer (not `jax_saic.centralized.
train()`) drives this — see that module's docstring for why
`jax_saic.centralized.train()` isn't reusable (it's hardcoded to
`jax_saic.env.step`, and its `select_action()`/`update()` close over
rendezvous-tuned constants like a fixed 5-per-agent action count and
`GAMMA=0.9`/`ALPHA=0.07`, none of which are safe to assume for a generic
task).

## 5. Diagnostics (per round)

`CertificationResult.round_stats`: fraction certified, `|F|`,
`min(m_eff)`, mean/median `U`, `delta_medoid`, cumulative extra env steps
consumed (tracked in real step units, not episode counts).

## 6. Function-approximation caveat

The ℓ∞ guarantee here is honest only in the tabular regime with a
controllable exploration floor (`mu_min > 0`). Under a neural value
function, `U(o)` from ensembles/counts becomes a *proxy*, not a bound —
off-distribution extrapolation has no sup-norm guarantee. The bracket
form's Monte Carlo lower bound (`V_pi_lower`) stays valid there (it's a
direct empirical rollout, not a theoretical bound); the optimistic upper
bound (`Q_ub`) does not.

## 7. Citations

Singh, S. & Yee, R. (1994). *An upper bound on the loss from approximate
optimal-value functions.* Machine Learning.
Williams, R. J. & Baird, L. C. (1993). *Tight performance bounds on
greedy policies based on imperfect value functions.*
Wainwright, M. J. (2019). *Variance-reduced Q-learning is minimax optimal.*
arXiv:1905.06265.
Li, G., Wei, Y., Chi, Y., Gu, Y. & Chen, Y. (2022). *Sample complexity of
asynchronous Q-learning: sharper analysis and variance reduction.* IEEE
Trans. Information Theory, 68(1):448–473.
Qu, G. & Wierman, A. (2020). *Finite-time analysis of asynchronous
stochastic approximation and Q-learning.* COLT.
Jin, C., Allen-Zhu, Z., Bubeck, S. & Jordan, M. I. (2018). *Is Q-learning
provably efficient?* NeurIPS.
Baird, L. C. (1995). *Residual algorithms: reinforcement learning with
function approximation.* ICML.
van Hasselt, H. (2010). *Double Q-learning.* NeurIPS.
van Hasselt, H., Guez, A. & Silver, D. (2016). *Deep reinforcement
learning with double Q-learning.* AAAI.

"""A GENERALIZABLE zero-variance value-of-observation, replacing BOTH the
Q-value noise AND the marginalization-weight noise of clustering.py's
value_of_observation() -- without injecting rendezvous-specific knowledge.

exact_value_of_observation.py (the earlier fix) removed Q-value noise
correctly (exact value iteration -- legitimate for any known, finite,
tabular MDP), but then replaced the marginalization weighting with an
ASSUMED uniform distribution over the other agent's position. That's two
different things conflated: it's true that both agents' STARTING positions
are i.i.d. uniform over non-goal cells (centralized.py's actual init code,
confirmed at jax_saic/centralized.py:217-221 -- legitimately "reading the
algorithm's own setup", not "solving rendezvous"). But N_table_emerged is
supposed to approximate the visitation distribution over the WHOLE
training trajectory under whatever policy is running, not just t=0 -- and
"assume it's uniform for the whole episode too" is a rendezvous-specific
shortcut (it happens to produce a clean symmetric answer on this grid, but
there's no general reason a differently-shaped multi-agent task's true
occupancy measure would stay uniform past the initial step).

The general fix: don't ASSUME any particular marginal shape. COMPUTE the
exact occupancy measure by forward-simulating the known, deterministic
transition dynamics (env.step, already available in ANY SAIC/ESAIC
application -- it's the environment model, same thing value iteration
itself already requires) under the exact GREEDY policy w.r.t. the exact Q*
(also already computed), starting from the known i.i.d.-uniform initial
distribution (again, just the algorithm's own init code, not
task-specific knowledge). Since both the dynamics and the converged greedy
policy are deterministic, every one of the finitely many legal starting
pairs traces out exactly ONE trajectory to termination -- no sampling, no
seed dependence, and the resulting occupancy is whatever it actually is
for the task at hand, uniform or not.

This generalizes to any tabular, finite, model-known multi-agent
coordination problem SAIC/ESAIC might be applied to -- it does not
presuppose rendezvous's symmetry, and would produce a non-uniform
occupancy in an environment where that's the true answer.
"""
from __future__ import annotations

import pathlib
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from check_value_precision import exact_value_iteration
from jax_saic.env import at_goal, step as env_step
from jax_saic.indexing import flatten_mixed_radix, unflatten_mixed_radix

NOA = 2


def exact_occupancy_value_of_observation(n: int, goal_idx: int, best_rew: float, gamma: float,
                                          step_cap: int | None = None, tie_tol: float = 1e-9):
    """Returns (V_o, N_o, residual_mass): the general, zero-variance
    value-of-observation. N_o is the EXACT occupancy measure (replacing
    N_table_emerged), derived by exact forward propagation, not assumption
    or sampling.

    IMPORTANT: single-path argmax rollout (breaking ties by lowest raw
    joint-action index) is NOT a valid general choice -- checked directly,
    73 of this grid's 81 joint states have multiple tied-optimal actions
    (moving toward the goal x-first vs y-first is equally optimal almost
    everywhere), and the raw action-index ordering (RIGHT,LEFT=0,1 before
    UP,DOWN=2,3) has no reason to respect the grid's diagonal symmetry.
    A single deterministic tie-break silently picks one arbitrary optimal
    trajectory per start state, which reintroduced spurious asymmetry
    between symmetric state pairs -- worse, in fact, than the thing being
    fixed. The general, tie-break-independent quantity is the occupancy
    measure under "follow any optimal action, ties split uniformly" --
    computed exactly via mass propagation (split probability mass equally
    across all tied-optimal actions at every step), not by picking one.
    """
    n2 = n * n
    goal_set0 = jnp.array([goal_idx])
    if step_cap is None:
        step_cap = 4 * n2  # generous; optimal_return.py shows true episodes need <= n (grid diag) steps

    Q = exact_value_iteration(n, goal_idx, best_rew, gamma)  # ((n^2)^2, 25)
    V_joint = Q.max(axis=-1)

    legit = [s for s in range(n2) if s != goal_idx]
    n_start = len(legit) ** NOA
    N_emerged = np.zeros(n2 * n2)  # exact joint-state occupancy, main_ps0-indexed (agent-0-fastest)

    # rho: probability mass at each joint state (main_ps0-indexed), under
    # the exact i.i.d.-uniform init (centralized.py:217-221) -- reading the
    # algorithm's own init code, not assuming a task-specific distribution.
    rho = np.zeros(n2 * n2)
    for s0 in legit:
        for s1 in legit:
            main_idx = int(flatten_mixed_radix(jnp.array([s0, s1]), base=n2))
            rho[main_idx] += 1.0 / n_start

    # Precompute, for every joint state, the set of tied-optimal joint
    # actions and the resulting next joint state for each.
    n_states = n2 * n2
    tied_next_states = [None] * n_states
    for s in range(n_states):
        ps0 = unflatten_mixed_radix(jnp.array(s), base=n2, k=NOA)
        if bool(jnp.any(at_goal(ps0, goal_set0))):
            tied_next_states[s] = []  # absorbing
            continue
        row = np.asarray(Q[s])
        tied_actions = np.flatnonzero(np.isclose(row, row.max(), atol=tie_tol))
        nexts = []
        for a in tied_actions:
            pa0 = unflatten_mixed_radix(jnp.array(int(a)), base=5, k=NOA)
            ps1, _err, _ter = env_step(ps0, pa0, n, goal_set0)
            nexts.append(int(flatten_mixed_radix(ps1, base=n2)))
        tied_next_states[s] = nexts

    for _t in range(step_cap):
        N_emerged += rho
        if rho.sum() < 1e-15:
            break
        new_rho = np.zeros(n2 * n2)
        for s in np.flatnonzero(rho > 0):
            nexts = tied_next_states[s]
            if not nexts:  # absorbing (at goal) -- mass already counted above, stops here
                continue
            share = rho[s] / len(nexts)
            for s1 in nexts:
                new_rho[s1] += share
        rho = new_rho
    residual_mass = rho.sum()

    # Marginalize exactly like clustering.value_of_observation(): V_o[i] is
    # agent-1's ("the last agent's") own position, weighted-averaged over
    # agent-0's position using the EXACT occupancy weights just computed.
    V_o = np.zeros(n2)
    N_o = np.zeros(n2)
    for i in range(n2):
        num, den = 0.0, 0.0
        for j in range(n2):
            main_idx = int(flatten_mixed_radix(jnp.array([j, i]), base=n2))
            w = N_emerged[main_idx]
            num += w * V_joint[main_idx]
            den += w
        V_o[i] = num / den if den > 0 else 0.0
        N_o[i] = den
    return V_o, N_o, residual_mass


if __name__ == "__main__":
    n = 3
    n2 = n * n
    goal_idx = n2 - 1
    best_rew = 10.0
    gamma = 0.9

    V_o, N_o, residual_mass = exact_occupancy_value_of_observation(n, goal_idx, best_rew, gamma)
    print(f"residual_mass (probability never absorbed at goal within step_cap): {residual_mass:.2e}")
    print("V_o (exact occupancy, general method):", np.round(V_o, 4))
    print("N_o (exact occupancy weights):", np.round(N_o, 4))

    print("\n=== Symmetric pairs (should be exactly equal, by the grid's own diagonal symmetry) ===")
    for a, b in [(1, 3), (2, 6), (5, 7)]:
        diff = abs(V_o[a] - V_o[b])
        print(f"states {a},{b}: V_o={V_o[a]:.6f}, {V_o[b]:.6f}  diff={diff:.2e}")

    from exact_value_of_observation import exact_value_of_observation
    V_o_uniform = exact_value_of_observation(n, goal_idx, best_rew, gamma)
    print("\n=== Compare to the earlier assumed-uniform-marginal version ===")
    print("V_o (assumed uniform):      ", np.round(V_o_uniform, 4))
    print("V_o (exact occupancy):      ", np.round(V_o, 4))
    print("max abs diff:", np.max(np.abs(V_o - V_o_uniform)))

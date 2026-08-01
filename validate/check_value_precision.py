"""Answers directly: once epsilon anneals toward 0 in the centralized
training phase, does value propagation continue for states off the
greedy policy's trajectory, or do their Q-value estimates go stale?

"The learned policy achieves near-optimal reward" (SS11.3) does NOT imply
"every state's value is precisely estimated" -- Q-learning only refines
Q(s,a) for (s,a) pairs actually visited, and once epsilon~0, an agent
mostly re-treads whatever trajectories the greedy policy already commits
to. This checks that gap directly rather than reasoning about it: since
the environment is fully known and deterministic (same fact that made
optimal_return.py's closed-form computation possible), the EXACT value
function can be computed via direct value iteration on the known MDP --
no Q-learning, no exploration, no epsilon, no sampling noise -- and
compared against what Q-learning actually produced.
"""
from __future__ import annotations

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from jax_saic.centralized import CentralizedConfig
from jax_saic.centralized import train as train_centralized
from jax_saic.env import step as env_step
from jax_saic.indexing import flatten_mixed_radix, unflatten_mixed_radix
from jax_saic import clustering

NOA = 2


def exact_value_iteration(n: int, goal_idx: int, best_rew: float, gamma: float, n_iters: int = 500):
    """Direct value iteration on the KNOWN, deterministic noa=2 MDP --
    the same transition/reward structure centralized.py learns via
    Q-learning, computed exactly instead. Returns Q* of shape
    ((n^2)^2, 5^2) and the derived V*(s) = max_a Q*(s,a).

    Reward structure matches centralized.py exactly (PORT_NOTES.md SS4.1):
    temp_rew = best_rew if ter==noa (full sync), 1 if ter>=1 (partial),
    0 otherwise -- this is the TRAINING reward (not the SS5.4/11 "reported"
    metric), since that's what the Q-values driving V_o are built from.
    """
    n2 = n * n
    n_states = n2**NOA
    n_actions = 5**NOA
    goal_set0 = jnp.array([goal_idx])

    # Precompute the deterministic transition + reward for every (s,a) pair once.
    next_state = np.zeros((n_states, n_actions), dtype=np.int64)
    reward = np.zeros((n_states, n_actions), dtype=np.float64)
    is_terminal_transition = np.zeros((n_states, n_actions), dtype=bool)

    for s in range(n_states):
        ps0 = unflatten_mixed_radix(jnp.array(s), base=n2, k=NOA)
        for a in range(n_actions):
            pa0 = unflatten_mixed_radix(jnp.array(a), base=5, k=NOA)
            ps1, _err, ter = env_step(ps0, pa0, n, goal_set0)
            s1 = int(flatten_mixed_radix(ps1, base=n2))
            next_state[s, a] = s1
            ter_int = int(ter)
            if ter_int == NOA:
                reward[s, a] = best_rew
                is_terminal_transition[s, a] = True
            elif ter_int >= 1:
                reward[s, a] = 1.0
                is_terminal_transition[s, a] = True
            else:
                reward[s, a] = 0.0

    Q = np.zeros((n_states, n_actions))
    for _ in range(n_iters):
        V_next = Q[next_state].max(axis=-1)  # (n_states, n_actions), V(s') for each (s,a)'s s'
        target = np.where(is_terminal_transition, reward, reward + gamma * V_next)
        if np.allclose(target, Q, atol=1e-10):
            Q = target
            break
        Q = target

    return Q


def main():
    n = 3
    n2 = n * n
    goal_idx = n2 - 1
    best_rew = 10.0
    gamma = 0.9
    inf_bits = 2

    print("Computing EXACT Q*/V* via value iteration on the known MDP...")
    Q_exact = exact_value_iteration(n, goal_idx, best_rew, gamma)

    print("\nRunning Q-learning-based centralized training (ns=80000, matching SS11.3)...")
    cfg = CentralizedConfig(n=n, goal_set0=jnp.array([goal_idx]), best_rew=best_rew, ns=80_000, end_learn=0.85)
    rng = jax.random.PRNGKey(0)
    qp_table, N_table_emerged, _rew = train_centralized(cfg, rng)
    V_o_learned, N_o_learned = clustering.value_of_observation(qp_table, N_table_emerged, n)
    print(f"V_o_learned = {np.round(V_o_learned, 4)}")
    print(f"N_o_learned (emerged-phase visitation) = {np.round(N_o_learned, 1)}")

    # Isolate Q-value precision specifically: reuse the SAME empirical
    # visitation weighting (N_table_emerged) the learned run actually
    # produced, only swapping in EXACT Q-values instead of learned ones.
    # If V_o differed only because of a different weighting scheme (e.g.
    # uniform vs visitation-based), that wouldn't isolate the thing being
    # asked about -- holding the weighting constant does.
    V_o_exact, _ = clustering.value_of_observation(Q_exact, N_table_emerged, n)
    print(f"V_o_exact (same empirical weighting, exact Q-values) = {np.round(V_o_exact, 4)}")

    print("\n=== Comparison: exact vs. Q-learned V_o (per state, index 0..8) ===")
    abs_err = np.abs(V_o_exact - V_o_learned)
    rel_err = abs_err / np.maximum(np.abs(V_o_exact), 1e-9)
    for i in range(n2):
        print(
            f"state {i}: exact={V_o_exact[i]:.4f}  learned={V_o_learned[i]:.4f}  "
            f"abs_err={abs_err[i]:.4f}  rel_err={100*rel_err[i]:.1f}%  N_o_emerged={N_o_learned[i]:.1f}"
        )

    # Does clustering assignment actually change if we cluster on the exact
    # values instead? Same N_o (weighting) both times -- only V_o differs.
    ag_learned = clustering.cluster_states(V_o_learned, N_o_learned, inf_bits, method="kmedian")
    ag_exact = clustering.cluster_states(V_o_exact, N_o_learned, inf_bits, method="kmedian")
    print("\nag_states from LEARNED V_o:")
    print(ag_learned)
    print("ag_states from EXACT V_o:")
    print(ag_exact)
    same = np.array_equal(np.sort(ag_learned, axis=1), np.sort(ag_exact, axis=1))
    print(f"Same clustering? {same}")


if __name__ == "__main__":
    main()

"""Fully deterministic, zero-variance replacement for
jax_saic.clustering.value_of_observation(), eliminating BOTH noise
sources identified in chat:

  1. Q-value noise: uses exact_value_iteration (check_value_precision.py)
     instead of Q-learning -- no sampling, no exploration, no epsilon.
  2. Marginalization-weight noise: the original formula (PORT_NOTES.md
     SS4.2) weights the average over the other agent's position using
     EMPIRICALLY SAMPLED visitation counts (N_table_emerged) -- a noisy,
     seed-dependent, limited-window (post-end_learn only) sample. But
     both agents' starting positions are drawn INDEPENDENTLY and
     UNIFORMLY over the same legal (non-goal) cells -- that's the actual,
     exactly-known initialization distribution, not something that needs
     to be estimated from a training run at all. So the exact marginal
     weight is 1/(n^2-1) for every other-agent position, in closed form.

Together these make V_o a deterministic function of (n, goal_idx,
best_rew, gamma) alone -- same answer on every seed, and should restore
the symmetry the grid's own diagonal-reflection geometry guarantees
(verified below, not just claimed).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from check_value_precision import exact_value_iteration
from jax_saic.indexing import flatten_mixed_radix, unflatten_mixed_radix

NOA = 2


def exact_value_of_observation(n: int, goal_idx: int, best_rew: float, gamma: float):
    """Returns V_o: shape (n^2,), the EXACT value of observation for
    "the last agent's" own position, marginalized uniformly over the
    other agent's position -- matching the actual i.i.d.-uniform,
    independent episode initialization exactly, not sampled from it.
    """
    n2 = n * n
    Q = exact_value_iteration(n, goal_idx, best_rew, gamma)  # ((n^2)^2, 25)
    V_joint = Q.max(axis=-1)  # (n^2)^2, V*(joint state)

    legit = [s for s in range(n2) if s != goal_idx]
    V_o = np.zeros(n2)
    for i in range(n2):
        vals = []
        for j in legit:  # other agent's position, uniform over legal cells
            ps0 = np.array([j, i])  # agent0=j, agent1=i (agent-1-fastest convention, indexing.py)
            main_ps0 = int(flatten_mixed_radix_np(ps0, n2))
            vals.append(V_joint[main_ps0])
        V_o[i] = np.mean(vals)
    return V_o


def flatten_mixed_radix_np(vals, base):
    # plain-numpy version of indexing.flatten_mixed_radix, avoiding a jax
    # array round-trip for a tight double loop.
    k = len(vals)
    powers = base ** np.arange(k)
    return int(np.sum(np.asarray(vals) * powers))


if __name__ == "__main__":
    n = 3
    n2 = n * n
    goal_idx = n2 - 1
    best_rew = 10.0
    gamma = 0.9

    V_o_exact_uniform = exact_value_of_observation(n, goal_idx, best_rew, gamma)
    print("Exact, zero-variance V_o (exact Q* + exact uniform marginalization):")
    print(np.round(V_o_exact_uniform, 4))

    print("\n=== Symmetric pairs (should now be EXACTLY equal) ===")
    for a, b in [(1, 3), (2, 6), (5, 7)]:
        diff = abs(V_o_exact_uniform[a] - V_o_exact_uniform[b])
        print(f"states {a},{b}: V_o={V_o_exact_uniform[a]:.6f}, {V_o_exact_uniform[b]:.6f}  diff={diff:.2e}")

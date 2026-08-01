"""Checks whether the V_o asymmetry between grid-symmetric state pairs
(e.g. states 5 and 7, both distance-1 from goal, mirror images across the
diagonal through the goal) is seed-specific noise (would shrink/average
out across independent centralized-training runs) or a systematic
property of the value-of-observation formula itself (persists regardless
of seed) -- raised directly in chat after Fig-8-style visualization
showed V_o[5]=2.64 vs V_o[7]=1.63, a 38% relative gap between states that
should be equivalent by the grid's own geometry.
"""
from __future__ import annotations

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from jax_saic import clustering
from jax_saic.centralized import CentralizedConfig
from jax_saic.centralized import train as train_centralized


def main():
    n = 3
    n2 = n * n
    goal_idx = n2 - 1
    goal_set0 = jnp.array([goal_idx])
    best_rew = 10.0
    ns_central = 80_000
    end_learn = 0.85

    n_seeds = 5
    all_V_o = []
    all_N_o = []

    for seed in range(n_seeds):
        cfg = CentralizedConfig(n=n, goal_set0=goal_set0, best_rew=best_rew, ns=ns_central, end_learn=end_learn)
        rng = jax.random.PRNGKey(seed)
        qp_table, N_table_emerged, rew = train_centralized(cfg, rng)
        V_o, N_o = clustering.value_of_observation(qp_table, N_table_emerged, n)
        all_V_o.append(np.asarray(V_o))
        all_N_o.append(np.asarray(N_o))
        steady_state = np.asarray(rew)[-len(rew) // 5:].mean()
        print(f"seed={seed}: steady_state_return={steady_state:.3f} V_o={np.round(V_o,3)}")

    all_V_o = np.stack(all_V_o)  # (n_seeds, 9)
    all_N_o = np.stack(all_N_o)

    print("\n=== Per-state V_o across seeds ===")
    for s in range(n2):
        vals = all_V_o[:, s]
        print(f"state {s}: mean={vals.mean():.3f} std={vals.std():.3f} values={np.round(vals,3)}")

    print("\n=== Symmetric pair (5,7) across seeds ===")
    for seed in range(n_seeds):
        v5, v7 = all_V_o[seed, 5], all_V_o[seed, 7]
        n5, n7 = all_N_o[seed, 5], all_N_o[seed, 7]
        print(f"seed={seed}: V_o[5]={v5:.3f} V_o[7]={v7:.3f} diff={abs(v5-v7):.3f} "
              f"({100*abs(v5-v7)/max(v5,v7):.0f}% rel)  N_o[5]={n5:.0f} N_o[7]={n7:.0f}")

    mean_V_o = all_V_o.mean(axis=0)
    print(f"\nAcross-seed MEAN V_o[5]={mean_V_o[5]:.3f}, V_o[7]={mean_V_o[7]:.3f}, "
          f"diff={abs(mean_V_o[5]-mean_V_o[7]):.3f} "
          f"({100*abs(mean_V_o[5]-mean_V_o[7])/max(mean_V_o[5],mean_V_o[7]):.0f}% rel)")

    print("\n=== Other symmetric pairs, across-seed mean ===")
    for a, b in [(1, 3), (2, 6)]:
        print(f"states {a},{b}: mean V_o = {mean_V_o[a]:.3f}, {mean_V_o[b]:.3f}, "
              f"diff={abs(mean_V_o[a]-mean_V_o[b]):.3f} "
              f"({100*abs(mean_V_o[a]-mean_V_o[b])/max(mean_V_o[a],mean_V_o[b]):.0f}% rel)")


if __name__ == "__main__":
    main()

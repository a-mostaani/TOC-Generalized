"""Runs decentralized training (noa=2,3,4) on top of the ag_states produced
by the EXACT, zero-variance value-of-observation (exact_value_of_observation.py
+ plot_grid_clusters_exact.py), instead of the normal empirically-clustered
ag_states -- raised directly in chat: "worth running decentralized training
phase on top of the obtained aggregation from this diagnostic tool, to see
if we get close to the optimal solutions or not -- if we don't, the problem
is in the decentralized training phase, as the aggregation is perfect here."

This isolates decentralized-training-logic correctness from clustering
precision: since the exact ag_states here is provably as good as it can be
for this environment (0.00e+00 symmetry, goal cell correctly high-valued),
any remaining shortfall vs. the closed-form optimal (optimal_return.py) at
noa=3/4 can no longer be blamed on clustering imprecision.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from jax_saic.train import DecentralizedConfig
from jax_saic.train import train as train_decentralized
from jax_saic import clustering

from exact_value_of_observation import exact_value_of_observation
from optimal_return import optimal_expected_reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noa", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--ns", type=int, nargs="+", default=[10_000, 300_000, 1_500_000],
                         help="decentralized ns per noa, same order as --noa (MATLAB defaults, PORT_NOTES.md SS8.1)")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    n = 3
    n2 = n * n
    inf_bits = 2
    goal_idx = n2 - 1
    goal_set0 = jnp.array([goal_idx])
    best_rew = 10.0
    gamma = 0.9
    end_learn_decentral = 0.80

    V_o = exact_value_of_observation(n, goal_idx, best_rew, gamma)
    N_o = np.ones(n2)
    ag_states = clustering.cluster_states(V_o, N_o, inf_bits, method="kmedian")
    print("Exact ag_states (0-indexed, -1=pad), used for ALL noa below:")
    print(np.asarray(ag_states))
    ag_states = jnp.asarray(ag_states)

    ns_by_noa = dict(zip(args.noa, args.ns if len(args.ns) == len(args.noa) else [args.ns[0]] * len(args.noa)))

    rng = jax.random.PRNGKey(0)
    print(f"\n{'noa':>4} {'jax steady-state':>20} {'optimal':>10} {'efficiency':>12}")
    for noa in args.noa:
        ns_dec = ns_by_noa[noa]
        tails = []
        for s in range(args.seeds):
            t0 = time.time()
            rng, rng_s = jax.random.split(rng)
            dcfg = DecentralizedConfig(
                n=n, noa=noa, inf_bits=inf_bits, goal_set0=goal_set0, best_rew=best_rew,
                gamma=gamma, ns=ns_dec, end_learn=end_learn_decentral,
            )
            rew, _ = train_decentralized(dcfg, ag_states, rng_s)
            rew = np.asarray(rew)
            tail = rew[-max(1, len(rew) // 5):].mean()
            tails.append(tail)
            print(f"  [noa={noa} seed={s}] steady-state={tail:.3f}  ({time.time()-t0:.1f}s, ns={ns_dec})")
        opt = optimal_expected_reward(n, noa, goal_idx, best_rew, gamma)
        mean_tail = np.mean(tails)
        eff = 100 * mean_tail / opt
        print(f"{noa:>4} {mean_tail:>12.3f} +/- {np.std(tails):<6.3f} {opt:>10.3f} {eff:>10.1f}%")

    print("\nDone. Compare these efficiency numbers to plot_normalized_return.py's "
          "figures (noa=2 ~94-99%, noa=3 ~53-61%, noa=4 ~12%, using the normal "
          "empirically-clustered ag_states) to see whether the exact aggregation "
          "closes the gap.")


if __name__ == "__main__":
    main()

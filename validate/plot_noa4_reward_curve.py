"""Captures the noa=4 decentralized-training reward-per-episode curve (on top
of the EXACT ag_states from exact_value_of_observation.py, same setup as
decentralized_on_exact_agstates.py) and plots it with a moving average,
per direct request -- decentralized_on_exact_agstates.py itself only prints
summary stats and discards the per-episode reward array.
"""
from __future__ import annotations

import pathlib
import sys

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from jax_saic.train import DecentralizedConfig
from jax_saic.train import train as train_decentralized
from jax_saic import clustering

from exact_value_of_observation import exact_value_of_observation

OUT_DIR = pathlib.Path(__file__).resolve().parent / "plots"


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or window > len(x):
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def main():
    n = 3
    n2 = n * n
    inf_bits = 2
    goal_idx = n2 - 1
    goal_set0 = jnp.array([goal_idx])
    best_rew = 10.0
    gamma = 0.9
    noa = 4
    ns = 1_500_000
    end_learn = 0.80
    window = 10_000

    V_o = exact_value_of_observation(n, goal_idx, best_rew, gamma)
    N_o = np.ones(n2)
    ag_states = clustering.cluster_states(V_o, N_o, inf_bits, method="kmedian")
    print("Exact ag_states (0-indexed, -1=pad):")
    print(np.asarray(ag_states))
    ag_states = jnp.asarray(ag_states)

    dcfg = DecentralizedConfig(
        n=n, noa=noa, inf_bits=inf_bits, goal_set0=goal_set0, best_rew=best_rew,
        gamma=gamma, ns=ns, end_learn=end_learn,
    )
    rng = jax.random.PRNGKey(0)
    print(f"Training noa={noa}, ns={ns} ...")
    rew, _ = train_decentralized(dcfg, ag_states, rng)
    rew = np.asarray(rew)

    raw_path = OUT_DIR.parent / "noa4_reward_curve.npy"
    np.save(raw_path, rew)
    print(f"Saved raw per-episode reward array to {raw_path}")

    smoothed = moving_average(rew, window)
    x = np.arange(len(smoothed)) + window // 2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, smoothed, lw=1.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward ({window}-episode moving average)")
    ax.set_title(f"noa={noa} decentralized training reward (exact ag_states)")
    fig.tight_layout()
    out_path = OUT_DIR / "noa4_reward_vs_episode_exact_agstates.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

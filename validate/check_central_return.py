"""Answers directly: what average return does the centralized-training
phase itself (always noa=2, PORT_NOTES.md SS0.8) actually converge to?
Checks whether the VoI signal feeding the clustering step comes from a
well-trained centralized Q-table, or a mediocre one -- since if the
centralized phase itself falls well short of optimal, that would explain
part of the downstream decentralized shortfall independent of the
decentralized coordination problem itself (SS11.2).
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
from optimal_return import optimal_expected_reward

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_CENTRAL = "#2a78d6"
COLOR_OPT = "#008300"
GRID = "#e1e0d9"
INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"

OUT_DIR = pathlib.Path(__file__).resolve().parent / "plots"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def moving_average(x, window):
    if window <= 1 or window > len(x):
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def main():
    n = 3
    goal_idx = n * n - 1
    goal_set0 = jnp.array([goal_idx])
    best_rew = 10.0
    gamma = 0.9
    ns = 80_000
    end_learn = 0.85

    cfg = CentralizedConfig(n=n, goal_set0=goal_set0, best_rew=best_rew, ns=ns, end_learn=end_learn)
    rng = jax.random.PRNGKey(0)
    qp_table, N_table_emerged, rew = train_centralized(cfg, rng)
    rew = np.asarray(rew)

    tail_len = max(1, len(rew) // 5)
    steady_state = rew[-tail_len:].mean()
    steady_state_std = rew[-tail_len:].std()
    early = rew[:tail_len].mean()

    optimal_noa2 = optimal_expected_reward(n, 2, goal_idx, best_rew, gamma)

    print(f"Centralized (noa=2) training, ns={ns}:")
    print(f"  early (first 20%) mean return: {early:.4f}")
    print(f"  steady-state (last 20%) mean return: {steady_state:.4f} +/- {steady_state_std:.4f}")
    print(f"  optimal (exact, no RL): {optimal_noa2:.4f}")
    print(f"  steady-state as % of optimal: {100*steady_state/optimal_noa2:.1f}%")

    OUT_DIR.mkdir(exist_ok=True)
    window = max(10, ns // 200)
    smoothed = moving_average(rew, window)
    x = np.arange(len(smoothed)) + window // 2

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style_axes(ax)
    ax.plot(x, smoothed, color=COLOR_CENTRAL, linewidth=2, label="centralized training (noa=2)", zorder=3)
    ax.axhline(optimal_noa2, color=COLOR_OPT, linewidth=1.5, linestyle="--", label=f"optimal ({optimal_noa2:.2f})", zorder=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward ({window}-episode moving average)")
    ax.set_title(f"Centralized training (noa=2) convergence, ns={ns}")
    legend = ax.legend(frameon=False, loc="lower right")
    for t in legend.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    out_path = OUT_DIR / "central_reward_vs_episode.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


if __name__ == "__main__":
    main()

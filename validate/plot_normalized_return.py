"""Puts the decentralized-training steady-state returns already measured
(medium-scale comparison, 10,000 episodes x 8 seeds, PORT_NOTES.md SS11)
into perspective by normalizing against the exact optimal expected return
(optimal_return.py -- computable in closed form, no RL needed, since the
environment is fully deterministic).

Produces:
  reward_vs_noa_with_optimal.png -- raw returns (JAX, Octave) with the
    optimal ceiling overlaid, per noa.
  efficiency_vs_noa.png -- the same data as a fraction of optimal (%),
    which is the more interpretable "how much of the achievable
    performance is captured" view: the optimal ceiling itself barely
    drops with noa, so this isolates how much harder DECENTRALIZED
    coordination gets as team size grows, separate from the task itself.
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimal_return import optimal_expected_reward

COLOR_JAX = "#2a78d6"
COLOR_REF = "#eb6834"
COLOR_OPT = "#008300"
GRID = "#e1e0d9"
INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"

OUT_DIR = pathlib.Path(__file__).resolve().parent / "plots"

# Medium-scale comparison results (PORT_NOTES.md SS11, "own independent
# clustering" -- the full end-to-end pipeline, not the anchored-ag_states
# ablation): 10,000 episodes, 8 seeds each side.
MEASURED = {
    2: {"jax_mean": 7.082, "jax_std": 0.031, "octave_mean": 7.455, "octave_std": 0.029},
    3: {"jax_mean": 3.834, "jax_std": 0.617, "octave_mean": 4.407, "octave_std": 0.221},
    4: {"jax_mean": 0.880, "jax_std": 0.010, "octave_mean": 0.838, "octave_std": 0.057},
}

N = 3
GOAL_IDX = N * N - 1
BEST_REW = 10.0
GAMMA = 0.9


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


def main():
    noas = sorted(MEASURED.keys())
    optimal = {noa: optimal_expected_reward(N, noa, GOAL_IDX, BEST_REW, GAMMA) for noa in noas}
    print("Optimal expected return (exact, no RL):", optimal)

    OUT_DIR.mkdir(exist_ok=True)

    # --- Plot 1: raw returns with optimal ceiling ---
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    _style_axes(ax)
    jax_means = [MEASURED[n]["jax_mean"] for n in noas]
    jax_stds = [MEASURED[n]["jax_std"] for n in noas]
    oct_means = [MEASURED[n]["octave_mean"] for n in noas]
    oct_stds = [MEASURED[n]["octave_std"] for n in noas]
    opt_vals = [optimal[n] for n in noas]

    ax.errorbar(noas, jax_means, yerr=jax_stds, color=COLOR_JAX, linewidth=2, marker="o",
                markersize=7, capsize=4, label="jax_saic (measured)", zorder=3)
    ax.errorbar(noas, oct_means, yerr=oct_stds, color=COLOR_REF, linewidth=2, marker="s",
                markersize=7, capsize=4, label="Octave reference (measured)", zorder=3)
    ax.plot(noas, opt_vals, color=COLOR_OPT, linewidth=2, linestyle="--", marker="^",
            markersize=8, label="optimal (exact, no RL)", zorder=3)

    ax.set_xlabel("Number of agents (noa)")
    ax.set_ylabel("Average return (last 20% of episodes)")
    ax.set_title("Decentralized return vs. the optimal ceiling")
    ax.set_xticks(noas)
    legend = ax.legend(frameon=False, loc="best")
    for t in legend.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    out1 = OUT_DIR / "reward_vs_noa_with_optimal.png"
    fig.savefig(out1, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] saved {out1}")

    # --- Plot 2: normalized efficiency (% of optimal) ---
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    _style_axes(ax)
    jax_eff = [100 * MEASURED[n]["jax_mean"] / optimal[n] for n in noas]
    jax_eff_std = [100 * MEASURED[n]["jax_std"] / optimal[n] for n in noas]
    oct_eff = [100 * MEASURED[n]["octave_mean"] / optimal[n] for n in noas]
    oct_eff_std = [100 * MEASURED[n]["octave_std"] / optimal[n] for n in noas]

    ax.errorbar(noas, jax_eff, yerr=jax_eff_std, color=COLOR_JAX, linewidth=2, marker="o",
                markersize=7, capsize=4, label="jax_saic", zorder=3)
    ax.errorbar(noas, oct_eff, yerr=oct_eff_std, color=COLOR_REF, linewidth=2, marker="s",
                markersize=7, capsize=4, label="Octave reference", zorder=3)
    ax.axhline(100, color=COLOR_OPT, linewidth=1.5, linestyle="--", label="optimal (100%)", zorder=2)

    ax.set_xlabel("Number of agents (noa)")
    ax.set_ylabel("Decentralized return as % of optimal")
    ax.set_title("Coordination efficiency vs. number of agents\n(isolates decentralized-learning difficulty from the near-flat task ceiling)")
    ax.set_xticks(noas)
    ax.set_ylim(0, 110)
    legend = ax.legend(frameon=False, loc="best")
    for t in legend.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    out2 = OUT_DIR / "efficiency_vs_noa.png"
    fig.savefig(out2, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] saved {out2}")

    print("\n=== Summary ===")
    for n in noas:
        print(
            f"noa={n}: optimal={optimal[n]:.3f} | "
            f"jax={MEASURED[n]['jax_mean']:.3f}+/-{MEASURED[n]['jax_std']:.3f} ({jax_eff[noas.index(n)]:.1f}% of optimal) | "
            f"octave={MEASURED[n]['octave_mean']:.3f}+/-{MEASURED[n]['octave_std']:.3f} ({oct_eff[noas.index(n)]:.1f}% of optimal)"
        )


if __name__ == "__main__":
    main()

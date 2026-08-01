"""Reproduces the visualization style of Fig. 8 in the SAIC paper
(arXiv:2005.14220): the n x n grid-world, each cell shaded by which
equivalence class (cluster) its observation belongs to after centralized
training + clustering. "Locations with similar colours... are grouped
into the same equivalence class" -- same idea, this port's noa=2
centralized phase + kmedian clustering (PORT_NOTES.md SS4.2/SS4.3),
matching the ns=80,000 run already used in SS11.3/SS11.4.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Categorical palette (dataviz skill reference, validated adjacent-pair
# CVD safety) -- one fixed color per cluster, in slot order.
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948", "#008300"]
INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"
GRID_LINE = "#c3c2b7"

OUT_DIR = pathlib.Path(__file__).resolve().parent / "plots"


def main():
    n = 3
    n2 = n * n
    goal_idx = n2 - 1
    goal_set0 = jnp.array([goal_idx])
    best_rew = 10.0
    gamma = 0.9
    inf_bits = 2
    ns_central = 80_000
    end_learn_central = 0.85

    print(f"Running centralized training (noa=2, ns={ns_central})...")
    cfg = CentralizedConfig(n=n, goal_set0=goal_set0, best_rew=best_rew, ns=ns_central, end_learn=end_learn_central)
    rng = jax.random.PRNGKey(0)
    qp_table, N_table_emerged, rew = train_centralized(cfg, rng)
    steady_state = np.asarray(rew)[-len(rew) // 5:].mean()
    print(f"Centralized steady-state return: {steady_state:.3f}")

    V_o, N_o = clustering.value_of_observation(qp_table, N_table_emerged, n)
    k = 2**inf_bits
    ag_states = clustering.cluster_states(V_o, N_o, inf_bits, method="kmedian")
    print("V_o:", np.round(V_o, 3))
    print("ag_states (0-indexed, -1=pad):")
    print(ag_states)

    # cluster_of[state] = cluster id
    cluster_of = np.full(n2, -1, dtype=int)
    for cid in range(k):
        for state in ag_states[cid]:
            if state >= 0:
                cluster_of[state] = cid

    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for state in range(n2):
        x, y = state % n, state // n  # 0-indexed coords, matching env.py's convention
        cid = cluster_of[state]
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor=GRID_LINE, linewidth=1.5))
        label = f"{state}\n(V={V_o[state]:.2f})"
        ax.text(x + 0.5, y + 0.5, label, ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        if state == goal_idx:
            ax.text(x + 0.5, y + 0.82, "G", ha="center", va="center", fontsize=16, color="white", fontweight="bold")

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend: one patch per cluster actually used, in fixed slot order.
    used_clusters = sorted(set(cluster_of.tolist()))
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)], edgecolor=GRID_LINE)
        for cid in used_clusters
    ]
    labels = [f"cluster {cid}" for cid in used_clusters]
    legend = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.03),
                        ncol=min(4, len(used_clusters)), frameon=False)
    for t in legend.get_texts():
        t.set_color(INK)

    ax.set_title(
        f"State aggregation (SAIC Fig. 8 style)\n"
        f"n={n}, R={inf_bits} bits ({k} equivalence classes), ns_central={ns_central}\n"
        f"centralized steady-state return: {steady_state:.2f}",
        color=INK, fontsize=11,
    )
    fig.tight_layout()
    out_path = OUT_DIR / "grid_clusters.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


if __name__ == "__main__":
    main()

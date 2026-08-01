"""Fig-8-style grid visualization using the EXACT, zero-variance
value-of-observation (exact_value_of_observation.py) instead of the
empirically-sampled one -- see chat: the empirical version showed a real,
non-shrinking asymmetry between grid-symmetric state pairs (e.g. states 5
and 7, both distance-1 from goal, showed a 38% relative V_o gap that
should be exactly zero by the grid's own diagonal-reflection symmetry).
This version has zero variance by construction (exact Q* + exact uniform
marginalization over the other agent's position, matching the actual
i.i.d.-uniform independent episode initialization exactly).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from exact_value_of_observation import exact_value_of_observation
from jax_saic import clustering

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
    best_rew = 10.0
    gamma = 0.9
    inf_bits = 2

    V_o = exact_value_of_observation(n, goal_idx, best_rew, gamma)
    N_o = np.ones(n2)
    k = 2**inf_bits
    ag_states = clustering.cluster_states(V_o, N_o, inf_bits, method="kmedian")
    print("V_o (exact):", np.round(V_o, 4))
    print("ag_states:")
    print(ag_states)

    cluster_of = np.full(n2, -1, dtype=int)
    for cid in range(k):
        for state in ag_states[cid]:
            if state >= 0:
                cluster_of[state] = cid

    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for state in range(n2):
        x, y = state % n, state // n
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
        f"State aggregation (SAIC Fig. 8 style) -- EXACT, zero-variance V_o\n"
        f"n={n}, R={inf_bits} bits ({k} equivalence classes)\n"
        f"exact Q* (value iteration) + exact uniform marginalization",
        color=INK, fontsize=11,
    )
    fig.tight_layout()
    out_path = OUT_DIR / "grid_clusters_exact.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


if __name__ == "__main__":
    main()

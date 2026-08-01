"""Investigates two things the repo owner asked about directly:

  1. How long does the centralized training phase (noa=2, always -- SS0.8)
     need to run before the resulting ag_states stabilizes enough to give
     good decentralized performance? Sweeps ns_central and, for each,
     reports diagnostics on V_o/N_o precision plus the resulting
     decentralized steady-state return.
  2. What does the original MATLAB "-50" indexing bug actually do in
     practice, and does it matter? At each ns_central, clusters BOTH via
     the resolved fix ("kmedian") and via a faithful replication of the
     original bug ("legacy_minus50", jax_saic.clustering's ablation mode),
     then runs decentralized training under each, to see whether the two
     labelings produce meaningfully different downstream performance.

Fixed at noa=3 (representative -- matches the ESAIC paper's own Fig. 6/7
agent count) for the decentralized comparison, to keep the sweep tractable.
"""
from __future__ import annotations

import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from jax_saic import clustering
from jax_saic.centralized import CentralizedConfig
from jax_saic.centralized import train as train_centralized
from jax_saic.train import DecentralizedConfig
from jax_saic.train import train as train_decentralized

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_FIX = "#2a78d6"
COLOR_LEGACY = "#e34948"
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


def main():
    n = 3
    n2 = n * n
    inf_bits = 2
    goal_set0 = jnp.array([n2 - 1])
    goal_idx = n2 - 1
    best_rew = 10.0
    gamma = 0.9
    end_learn_central = 0.85
    end_learn_decentral = 0.80

    ns_central_list = [1000, 5000, 20000, 80000]
    noa = 3
    ns_decentral = 10_000
    n_seeds = 4

    rng = jax.random.PRNGKey(0)

    rows = []
    for ns_central in ns_central_list:
        print(f"\n=== ns_central={ns_central} ===")
        t0 = time.time()
        rng, rng_c = jax.random.split(rng)
        cfg = CentralizedConfig(n=n, goal_set0=goal_set0, best_rew=best_rew, ns=ns_central, end_learn=end_learn_central)
        qp_c, N_emerged_c, _central_rew = train_centralized(cfg, rng_c)
        print(f"[central] ns={ns_central} done in {time.time()-t0:.1f}s")

        V_o, N_o = clustering.value_of_observation(qp_c, N_emerged_c, n)
        print(f"V_o={np.round(V_o,3)}")
        print(f"N_o={np.round(N_o,3)}")
        print(f"N_o[goal={goal_idx}]={N_o[goal_idx]:.4f}, min(N_o)={N_o.min():.4f} at state {int(np.argmin(N_o))}, "
              f"states with N_o<50: {int(np.sum(N_o < 50))}/{n2}")

        ag_fixed = clustering.cluster_states(V_o, N_o, inf_bits, method="kmedian")
        ag_legacy, diag = clustering.cluster_states(
            V_o, N_o, inf_bits, method="legacy_minus50", rng=np.random.default_rng(0), return_diagnostics=True
        )
        print(f"legacy_minus50: crashed_states={diag['crashed_states']}, misassigned_states={diag['misassigned_states']}")
        same_ag = np.array_equal(np.sort(ag_fixed, axis=1), np.sort(ag_legacy, axis=1))
        print(f"ag_states identical (fixed vs legacy)? {same_ag}")

        row = {
            "ns_central": ns_central,
            "N_o_goal": float(N_o[goal_idx]),
            "N_o_min": float(N_o.min()),
            "n_understaffed": int(np.sum(N_o < 50)),
            "crashed_states": diag["crashed_states"],
            "misassigned_states": diag["misassigned_states"],
            "ag_states_identical": bool(same_ag),
        }

        for method_name, ag in (("kmedian_fix", ag_fixed), ("legacy_minus50", ag_legacy)):
            curves = []
            for s in range(n_seeds):
                t0 = time.time()
                rng, rng_s = jax.random.split(rng)
                dcfg = DecentralizedConfig(
                    n=n, noa=noa, inf_bits=inf_bits, goal_set0=goal_set0, best_rew=best_rew,
                    gamma=gamma, ns=ns_decentral, end_learn=end_learn_decentral,
                )
                rew, _ = train_decentralized(dcfg, jnp.asarray(ag), rng_s)
                curves.append(np.asarray(rew))
                print(f"[decentral {method_name}] ns_central={ns_central} seed={s} done in {time.time()-t0:.1f}s")
            tails = [c[-max(1, len(c) // 5):].mean() for c in curves]
            row[f"{method_name}_mean"] = float(np.mean(tails))
            row[f"{method_name}_std"] = float(np.std(tails))
            row[f"{method_name}_curves"] = curves

        rows.append(row)

    OUT_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style_axes(ax)
    xs = [r["ns_central"] for r in rows]
    fix_means = [r["kmedian_fix_mean"] for r in rows]
    fix_stds = [r["kmedian_fix_std"] for r in rows]
    legacy_means = [r["legacy_minus50_mean"] for r in rows]
    legacy_stds = [r["legacy_minus50_std"] for r in rows]
    ax.errorbar(xs, fix_means, yerr=fix_stds, color=COLOR_FIX, linewidth=2, marker="o", markersize=7,
                capsize=4, label="kmedian fix (nearest-medoid)", zorder=3)
    ax.errorbar(xs, legacy_means, yerr=legacy_stds, color=COLOR_LEGACY, linewidth=2, marker="s", markersize=7,
                capsize=4, label="legacy -50 trick", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Centralized training episodes (ns_central, log scale)")
    ax.set_ylabel(f"Decentralized steady-state return (noa={noa})")
    ax.set_title("Effect of centralized-training length and clustering method on decentralized performance")
    legend = ax.legend(frameon=False, loc="best")
    for t in legend.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    out_path = OUT_DIR / "sweep_central_length.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\n[plot] saved {out_path}")

    print("\n=== Summary ===")
    for r in rows:
        print(
            f"ns_central={r['ns_central']:>7d} | N_o[goal]={r['N_o_goal']:.3f} min(N_o)={r['N_o_min']:.3f} "
            f"understaffed={r['n_understaffed']}/{n2} | ag identical={r['ag_states_identical']} "
            f"misassigned={r['misassigned_states']} | "
            f"kmedian_fix={r['kmedian_fix_mean']:.3f}+/-{r['kmedian_fix_std']:.3f} "
            f"legacy_minus50={r['legacy_minus50_mean']:.3f}+/-{r['legacy_minus50_std']:.3f}"
        )


if __name__ == "__main__":
    main()

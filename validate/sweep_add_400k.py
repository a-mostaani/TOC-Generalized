"""Adds one more point to the SS11.1 centralized-training-length sweep:
ns_central=400,000 (5x the previous largest point, 80,000), same
methodology (validate/sweep_central_length.py) -- centralized training at
noa=2, value-of-observation, clustering both ways ("kmedian" fix vs.
"legacy_minus50" ablation), decentralized training at noa=3 for 10,000
episodes x 4 seeds under each.

Re-plots the full sweep (this point plus the four already computed and
logged from the earlier run, hardcoded below rather than re-run --
recomputing them would cost the same ~35 min they already took for no new
information).
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

# From the earlier sweep's logged summary (validate/sweep_central_length.py,
# PORT_NOTES.md SS11.1) -- not re-run, just carried forward for the combined plot.
PREVIOUS_POINTS = [
    {"ns_central": 1000, "kmedian_fix_mean": 3.997, "kmedian_fix_std": 0.428,
     "legacy_minus50_mean": 6.277, "legacy_minus50_std": 0.542},
    {"ns_central": 5000, "kmedian_fix_mean": 4.156, "kmedian_fix_std": 0.250,
     "legacy_minus50_mean": 4.149, "legacy_minus50_std": 0.623},
    {"ns_central": 20000, "kmedian_fix_mean": 4.133, "kmedian_fix_std": 0.389,
     "legacy_minus50_mean": 4.024, "legacy_minus50_std": 0.418},
    {"ns_central": 80000, "kmedian_fix_mean": 4.695, "kmedian_fix_std": 0.290,
     "legacy_minus50_mean": 4.290, "legacy_minus50_std": 0.472},
]


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

    ns_central = 400_000
    noa = 3
    ns_decentral = 10_000
    n_seeds = 4

    rng = jax.random.PRNGKey(0)

    print(f"=== ns_central={ns_central} ===")
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

    row = {"ns_central": ns_central}
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

    all_rows = PREVIOUS_POINTS + [row]
    all_rows.sort(key=lambda r: r["ns_central"])

    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style_axes(ax)
    xs = [r["ns_central"] for r in all_rows]
    fix_means = [r["kmedian_fix_mean"] for r in all_rows]
    fix_stds = [r["kmedian_fix_std"] for r in all_rows]
    legacy_means = [r["legacy_minus50_mean"] for r in all_rows]
    legacy_stds = [r["legacy_minus50_std"] for r in all_rows]
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
    print(f"\n[plot] saved (updated, now {len(all_rows)} points) {out_path}")

    print("\n=== Full summary (all points) ===")
    for r in all_rows:
        print(
            f"ns_central={r['ns_central']:>7d} | "
            f"kmedian_fix={r['kmedian_fix_mean']:.3f}+/-{r['kmedian_fix_std']:.3f} "
            f"legacy_minus50={r['legacy_minus50_mean']:.3f}+/-{r['legacy_minus50_std']:.3f}"
        )


if __name__ == "__main__":
    main()

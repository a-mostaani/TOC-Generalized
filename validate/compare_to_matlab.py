"""Validate jax_saic against the reference MATLAB/Octave algorithm.

Runs jax_saic's full Algorithm 1 pipeline (centralized training, always at
noa=2 per ESAIC Theorem 1 -- PORT_NOTES.md SS0.8 -- then clustering, then
decentralized training at whatever noa the comparison needs) across
several seeds, and compares against reference curves produced by actually
running the original algorithm files under GNU Octave
(validate/octave_run_centralized.m, validate/octave_run_decentralized.m --
see those files and PORT_NOTES.md for the compatibility shims involved).

Per the task spec: do NOT expect bit-exact equivalence (MATLAB/Octave and
JAX use different RNGs). The target is that reward-curve shape,
convergence episode count, and steady-state reward level match within
reasonable variance across seeds.

Produces two plots (per the repo owner's request, matching the style of
Figures 6 and 7 of the ESAIC paper):
  1. reward_vs_episode.png -- moving-average reward vs. episode, JAX
     (mean +/- range across seeds) vs. the Octave reference curve, for one
     representative noa.
  2. reward_vs_noa.png -- post-convergence (steady-state) average return
     vs. number of agents, JAX (mean +/- std across seeds) vs. Octave
     reference, swept over noa.

Usage:
  python validate/compare_to_matlab.py --scale small   # ~2-3k episodes, 3 seeds (default)
  python validate/compare_to_matlab.py --scale full    # MATLAB-comment-suggested episode counts, 10 seeds
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from jax_saic import clustering
from jax_saic.centralized import CentralizedConfig
from jax_saic.centralized import train as train_centralized
from jax_saic.train import DecentralizedConfig
from jax_saic.train import train as train_decentralized

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical palette slots 1 (JAX) and 2 (reference) from the dataviz
# skill's validated default -- an adjacent pair already cleared for CVD
# safety (worst adjacent CVD dE 9.1 light, >=8 target), not re-validated
# here since only this one fixed pair is used.
COLOR_JAX = "#2a78d6"
COLOR_JAX_BAND = "#9ec5f4"
COLOR_REF = "#eb6834"
GRID = "#e1e0d9"
INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"

WORK_DIR = pathlib.Path(__file__).resolve().parent / "octave_work"
OUT_DIR = pathlib.Path(__file__).resolve().parent / "plots"


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or window > len(x):
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def run_jax_pipeline(
    n: int,
    goal_set0: jnp.ndarray,
    best_rew: float,
    gamma: float,
    inf_bits: int,
    ns_central: int,
    end_learn_central: float,
    noa_list: list[int],
    ns_decentral: int,
    end_learn_decentral: float,
    n_seeds: int,
    seed0: int = 0,
):
    """One centralized+clustering run (noa=2, shared across every noa in
    noa_list per ESAIC Theorem 1), then n_seeds decentralized runs at each
    noa. Returns {noa: [reward_array_per_seed, ...]}.
    """
    print(f"[jax] centralized training (noa=2, ns={ns_central})...")
    t0 = time.time()
    central_cfg = CentralizedConfig(
        n=n, goal_set0=goal_set0, best_rew=best_rew, ns=ns_central, end_learn=end_learn_central
    )
    rng = jax.random.PRNGKey(seed0)
    rng, rng_central = jax.random.split(rng)
    qp_central, N_emerged_central, _central_rew = train_centralized(central_cfg, rng_central)
    print(f"[jax] centralized training done in {time.time()-t0:.1f}s")

    V_o, N_o = clustering.value_of_observation(qp_central, N_emerged_central, n)
    ag_states_np = clustering.cluster_states(V_o, N_o, inf_bits, method="kmedian")
    ag_states = jnp.asarray(ag_states_np)
    print(f"[jax] ag_states (0-indexed, -1=pad):\n{ag_states_np}")

    results = {}
    for noa in noa_list:
        curves = []
        for s in range(n_seeds):
            t0 = time.time()
            rng, rng_s = jax.random.split(rng)
            dcfg = DecentralizedConfig(
                n=n,
                noa=noa,
                inf_bits=inf_bits,
                goal_set0=goal_set0,
                best_rew=best_rew,
                gamma=gamma,
                ns=ns_decentral,
                end_learn=end_learn_decentral,
            )
            rew, _ = train_decentralized(dcfg, ag_states, rng_s)
            curves.append(np.asarray(rew))
            print(f"[jax] noa={noa} seed={s} done in {time.time()-t0:.1f}s")
        results[noa] = curves
    return results


def load_octave_reference(noa_list: list[int]):
    """Returns {noa: [reward_array_per_seed, ...]}. Prefers the multiseed
    sweep (octave_work/multiseed/decentral_noa{N}_seed{S}.mat) if present
    -- falls back to the single-seed reference file otherwise."""
    ref = {}
    multiseed_dir = WORK_DIR / "multiseed"
    for noa in noa_list:
        curves = []
        if multiseed_dir.exists():
            for f in sorted(multiseed_dir.glob(f"decentral_noa{noa}_seed*.mat")):
                d = scipy.io.loadmat(f)
                curves.append(np.asarray(d["rew"]).flatten())
        if not curves:
            f = WORK_DIR / f"decentral_reference_noa{noa}.mat"
            if f.exists():
                d = scipy.io.loadmat(f)
                curves.append(np.asarray(d["rew"]).flatten())
        if curves:
            ref[noa] = curves
            print(f"[octave] noa={noa}: loaded {len(curves)} reference seed(s)")
        else:
            print(f"[warn] no Octave reference found for noa={noa}")
    return ref


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


def plot_reward_vs_episode(jax_curves_by_noa, ref_by_noa, noa, window, out_path):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style_axes(ax)

    curves = jax_curves_by_noa.get(noa, [])
    if curves:
        smoothed = [moving_average(c, window) for c in curves]
        min_len = min(len(s) for s in smoothed)
        stacked = np.stack([s[:min_len] for s in smoothed])
        mean = stacked.mean(axis=0)
        lo, hi = stacked.min(axis=0), stacked.max(axis=0)
        x = np.arange(min_len) + window // 2
        ax.fill_between(x, lo, hi, color=COLOR_JAX_BAND, alpha=0.5, linewidth=0, zorder=2)
        ax.plot(x, mean, color=COLOR_JAX, linewidth=2, label=f"jax_saic (n={len(curves)} seeds)", zorder=3)

    ref_curves = ref_by_noa.get(noa, [])
    if ref_curves:
        ref_smoothed = [moving_average(c, window) for c in ref_curves]
        min_len = min(len(s) for s in ref_smoothed)
        ref_stacked = np.stack([s[:min_len] for s in ref_smoothed])
        ref_mean = ref_stacked.mean(axis=0)
        x_ref = np.arange(min_len) + window // 2
        if len(ref_curves) > 1:
            ref_lo, ref_hi = ref_stacked.min(axis=0), ref_stacked.max(axis=0)
            ax.fill_between(x_ref, ref_lo, ref_hi, color=COLOR_REF, alpha=0.2, linewidth=0, zorder=1)
        ax.plot(
            x_ref, ref_mean, color=COLOR_REF, linewidth=2,
            label=f"Octave reference (n={len(ref_curves)} seeds)", zorder=3,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward ({window}-episode moving average)")
    ax.set_title(f"Reward vs. episode -- decentralized training, noa={noa}")
    legend = ax.legend(frameon=False, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_reward_vs_noa(jax_curves_by_noa, ref_by_noa, tail_frac, out_path):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    _style_axes(ax)

    noas = sorted(jax_curves_by_noa.keys())
    jax_means, jax_stds = [], []
    for noa in noas:
        tails = []
        for c in jax_curves_by_noa[noa]:
            tail_len = max(1, int(len(c) * tail_frac))
            tails.append(c[-tail_len:].mean())
        jax_means.append(np.mean(tails))
        jax_stds.append(np.std(tails))

    ax.errorbar(
        noas, jax_means, yerr=jax_stds, color=COLOR_JAX, linewidth=2, marker="o",
        markersize=7, capsize=4, label="jax_saic (mean +/- std across seeds)", zorder=3,
    )

    ref_noas, ref_means, ref_stds = [], [], []
    for noa in noas:
        if noa in ref_by_noa:
            tails = []
            for c in ref_by_noa[noa]:
                tail_len = max(1, int(len(c) * tail_frac))
                tails.append(c[-tail_len:].mean())
            ref_noas.append(noa)
            ref_means.append(np.mean(tails))
            ref_stds.append(np.std(tails))
    if ref_means:
        ax.errorbar(
            ref_noas, ref_means, yerr=ref_stds, color=COLOR_REF, linewidth=2, marker="s",
            markersize=7, capsize=4, label="Octave reference (mean +/- std across seeds)", zorder=3,
        )

    ax.set_xlabel("Number of agents (noa)")
    ax.set_ylabel(f"Post-convergence average return (last {int(tail_frac*100)}% of episodes)")
    ax.set_title("Post-convergence average return vs. number of agents")
    ax.set_xticks(noas)
    legend = ax.legend(frameon=False, loc="best")
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["small", "medium", "full"], default="small")
    parser.add_argument("--noa", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--fig6-noa", type=int, default=3, help="Which noa to use for the reward-vs-episode plot")
    args = parser.parse_args()

    n = 3
    inf_bits = 2
    goal_set0 = jnp.array([n * n - 1])
    best_rew = 10.0
    gamma = 0.9
    end_learn_central = 0.85
    end_learn_decentral = 0.80

    if args.scale == "small":
        ns_central = 3000
        ns_decentral = 3000
        n_seeds = args.seeds or 3
    elif args.scale == "medium":
        ns_central = 3000  # centralized phase is noa=2-only and already converges fast; unchanged
        ns_decentral = 10_000
        n_seeds = args.seeds or 8
    else:
        ns_central = 120_000
        ns_decentral = 300_000  # MATLAB comment default for noa=3, n=3
        n_seeds = args.seeds or 10

    OUT_DIR.mkdir(exist_ok=True)

    jax_curves = run_jax_pipeline(
        n=n,
        goal_set0=goal_set0,
        best_rew=best_rew,
        gamma=gamma,
        inf_bits=inf_bits,
        ns_central=ns_central,
        end_learn_central=end_learn_central,
        noa_list=args.noa,
        ns_decentral=ns_decentral,
        end_learn_decentral=end_learn_decentral,
        n_seeds=n_seeds,
    )
    ref_curves = load_octave_reference(args.noa)

    window = max(10, ns_decentral // 100)
    plot_reward_vs_episode(
        jax_curves, ref_curves, args.fig6_noa, window, OUT_DIR / "reward_vs_episode.png"
    )
    plot_reward_vs_noa(jax_curves, ref_curves, tail_frac=0.2, out_path=OUT_DIR / "reward_vs_noa.png")

    print("\n=== Summary ===")
    for noa in args.noa:
        jax_tail = [c[-max(1, len(c) // 5):].mean() for c in jax_curves.get(noa, [])]
        ref_tail = [c[-max(1, len(c) // 5):].mean() for c in ref_curves.get(noa, [])]
        print(
            f"noa={noa}: jax steady-state = {np.mean(jax_tail):.3f} +/- {np.std(jax_tail):.3f}"
            f" (n={len(jax_tail)} seeds), "
            f"octave reference = {np.mean(ref_tail):.3f} +/- {np.std(ref_tail):.3f} (n={len(ref_tail)} seeds)"
        )


if __name__ == "__main__":
    main()

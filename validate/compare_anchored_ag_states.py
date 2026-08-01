"""Controlled experiment: decentralized training only, both sides anchored
to the SAME ag_states (the Octave-generated one), to isolate whether the
noa=3 gap seen in compare_to_matlab.py comes from the decentralized
training itself or from upstream differences in the independently-run
centralized training + clustering (each side normally does its own,
seeded differently -- a real confound the repo owner flagged).

The Octave multiseed reference curves (validate/octave_work/multiseed/)
already all share ONE ag_states (whatever
validate/octave_work/central_reference.mat's centralized run produced --
every Octave decentralized seed loads the same
agreggated_states_n3_g9_infbits2_realSAIC.mat). This script makes jax_saic
use that EXACT SAME ag_states (converted from MATLAB's 1-indexed/0-pad
convention to this port's 0-indexed/-1-pad convention -- PORT_NOTES.md
SS4.3) instead of clustering its own, then compares.

Usage:
  python validate/compare_anchored_ag_states.py --ns 10000 --seeds 8
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

from jax_saic.train import DecentralizedConfig
from jax_saic.train import train as train_decentralized

from compare_to_matlab import (
    COLOR_JAX,
    COLOR_JAX_BAND,
    COLOR_REF,
    OUT_DIR,
    WORK_DIR,
    _style_axes,
    load_octave_reference,
    moving_average,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_octave_ag_states(n2: int) -> jnp.ndarray:
    """Loads central_reference.mat's ag_states_median (MATLAB 1-indexed,
    0=pad) and converts to this port's 0-indexed, -1=pad convention."""
    d = scipy.io.loadmat(WORK_DIR / "central_reference.mat")
    ag = np.asarray(d["ag_states_median"]).astype(np.int32)
    ag0 = np.where(ag == 0, -1, ag - 1)
    return jnp.asarray(ag0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noa", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--ns", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--fig6-noa", type=int, default=3)
    args = parser.parse_args()

    n = 3
    inf_bits = 2
    goal_set0 = jnp.array([n * n - 1])
    best_rew = 10.0
    gamma = 0.9
    end_learn_decentral = 0.80

    ag_states = load_octave_ag_states(n * n)
    print("Anchored ag_states (0-indexed, -1=pad), shared by BOTH sides:")
    print(np.asarray(ag_states))

    rng = jax.random.PRNGKey(0)
    jax_curves = {}
    for noa in args.noa:
        curves = []
        for s in range(args.seeds):
            t0 = time.time()
            rng, rng_s = jax.random.split(rng)
            dcfg = DecentralizedConfig(
                n=n, noa=noa, inf_bits=inf_bits, goal_set0=goal_set0, best_rew=best_rew,
                gamma=gamma, ns=args.ns, end_learn=end_learn_decentral,
            )
            rew, _ = train_decentralized(dcfg, ag_states, rng_s)
            curves.append(np.asarray(rew))
            print(f"[jax-anchored] noa={noa} seed={s} done in {time.time()-t0:.1f}s")
        jax_curves[noa] = curves

    ref_curves = load_octave_reference(args.noa)

    OUT_DIR.mkdir(exist_ok=True)
    window = max(10, args.ns // 100)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
    _style_axes(ax)
    noa = args.fig6_noa
    curves = jax_curves.get(noa, [])
    if curves:
        smoothed = [moving_average(c, window) for c in curves]
        min_len = min(len(s) for s in smoothed)
        stacked = np.stack([s[:min_len] for s in smoothed])
        x = np.arange(min_len) + window // 2
        ax.fill_between(x, stacked.min(axis=0), stacked.max(axis=0), color=COLOR_JAX_BAND, alpha=0.5, linewidth=0, zorder=2)
        ax.plot(x, stacked.mean(axis=0), color=COLOR_JAX, linewidth=2, label=f"jax_saic, anchored ag_states (n={len(curves)} seeds)", zorder=3)
    ref = ref_curves.get(noa, [])
    if ref:
        ref_smoothed = [moving_average(c, window) for c in ref]
        min_len = min(len(s) for s in ref_smoothed)
        ref_stacked = np.stack([s[:min_len] for s in ref_smoothed])
        x_ref = np.arange(min_len) + window // 2
        if len(ref) > 1:
            ax.fill_between(x_ref, ref_stacked.min(axis=0), ref_stacked.max(axis=0), color=COLOR_REF, alpha=0.2, linewidth=0, zorder=1)
        ax.plot(x_ref, ref_stacked.mean(axis=0), color=COLOR_REF, linewidth=2, label=f"Octave reference (n={len(ref)} seeds)", zorder=3)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward ({window}-episode moving average)")
    ax.set_title(f"Anchored-ag_states comparison, noa={noa} (same ag_states both sides)")
    legend = ax.legend(frameon=False, loc="lower right")
    for t in legend.get_texts():
        t.set_color("#0b0b0b")
    fig.tight_layout()
    out_path = OUT_DIR / "reward_vs_episode_anchored.png"
    fig.savefig(out_path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"[plot] saved {out_path}")

    print("\n=== Anchored-ag_states summary (same ag_states fed to both sides) ===")
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

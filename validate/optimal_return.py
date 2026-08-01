"""Exact, RL-free computation of the optimal expected reward for the
geometric-consensus rendezvous task, used as a normalization ceiling for
decentralized-training steady-state returns (see chat -- the repo owner's
observation that this is computable without running the centralized
training phase, since the environment is fully deterministic).

Derivation (CORRECTED once already -- see git history / chat: an earlier
version of this reasoning wrongly assumed padding an agent's arrival time
requires a "there and back" detour, hence only even-parity delays were
reachable, and split into a parity-matched/parity-mismatched case. That
missed the STAY action entirely: since STAY leaves position unchanged, an
agent can delay its arrival by exactly 1 step at a time with NO parity
restriction, as long as the padding happens BEFORE it ever touches the
goal cell (touching goal ends the episode immediately, so padding can't
happen by visiting-and-leaving). This was caught by the corrected formula
producing implausible numbers -- lower than actually-measured trained
performance, which is impossible for a true upper bound -- not by
reasoning alone.

Corrected derivation (verified against a STAY-based scripted simulator in
verify_against_simulation() below, not just reasoned through):

  - The environment is deterministic: env.step(ps0, pa0) always returns
    the same next state. Episodes end the instant ANY agent reaches the
    goal cell (confirmed empirically in PORT_NOTES.md SS10.1), and the
    REPORTED reward metric (rew[i] -- not the training-time temp_rew,
    PORT_NOTES.md SS5.4) is a step function of how many agents were
    present at that moment:
        all noa agents  -> best_rew * gamma**T
        some (not all)  -> 1        * gamma**T
        none (stuck)    -> 0
    where T is the number of environment steps taken (counter(i) = T+1 at
    episode-end, verified directly by mirroring train.py's exact
    counter/break bookkeeping on a scripted trajectory, not just traced
    by hand).

  - Because STAY allows exactly-1-step delay with no parity restriction,
    EVERY agent can arrive at ANY T >= its own shortest distance d_i (pad
    with STAY moves before reaching goal, then walk in on the final step).
    So full-team simultaneous arrival is ALWAYS achievable -- there is no
    "parities differ" failure case at all. The optimal policy is simply:
    every agent takes its shortest path, padding with STAY as needed, so
    all arrive together at T = max_i(d_i). Optimal reward = best_rew *
    gamma**max_i(d_i), unconditionally.

  - Averaging over all (n^2-1)^noa equally-likely starting tuples (i.i.d.
    uniform over non-goal cells, with replacement -- matching how episodes
    are actually initialized, train.py) gives the exact expected optimal
    return. Enumerated exactly (512 tuples for n=3, noa=3), not sampled.
"""
from __future__ import annotations

import itertools

import numpy as np


def _distances(n: int, goal_idx: int) -> dict[int, int]:
    gx, gy = goal_idx % n, goal_idx // n
    return {s: abs(s % n - gx) + abs(s // n - gy) for s in range(n * n) if s != goal_idx}


def optimal_expected_reward(n: int, noa: int, goal_idx: int, best_rew: float, gamma: float) -> float:
    """Full synchronization is always achievable (STAY gives 1-step,
    parity-free padding -- see module docstring), so the optimal reward
    for any starting tuple is simply best_rew * gamma**max_i(d_i)."""
    dist = _distances(n, goal_idx)
    legit = sorted(dist.keys())
    total = 0.0
    count = 0
    for combo in itertools.product(legit, repeat=noa):
        T = max(dist[s] for s in combo)
        total += best_rew * (gamma**T)
        count += 1
    return total / count


def verify_against_simulation(n: int, noa: int, goal_idx: int, best_rew: float, gamma: float, n_trials: int = 200):
    """Cross-check the closed-form formula against literally simulating a
    hand-scripted (non-learned) optimal policy through env.step, for random
    starting configurations. Not exhaustive -- a spot check that the
    reasoning above is right, not just internally consistent."""
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    import jax.numpy as jnp
    from jax_saic.env import step as env_step, RIGHT, LEFT, UP, DOWN, STAY

    dist = _distances(n, goal_idx)
    legit = sorted(dist.keys())
    goal_set0 = jnp.array([goal_idx])
    rng = np.random.default_rng(0)

    mismatches = []
    for _ in range(n_trials):
        combo = tuple(int(rng.choice(legit)) for _ in range(noa))
        ds = [dist[s] for s in combo]
        target_T = max(ds)  # full sync always achievable (STAY padding, no parity restriction)

        # Script each agent's exact path: pad FIRST with STAY (1-step
        # increments, never touching goal -- touching goal ends the
        # episode immediately, so padding must happen strictly before
        # arrival), THEN greedy Manhattan descent straight to goal,
        # arriving fresh exactly on step target_T.
        ps0 = jnp.array(combo)
        plans = []
        for s in combo:
            x, y = s % n, s // n
            gx, gy = goal_idx % n, goal_idx // n
            d_i = dist[s]
            pad_steps = target_T - d_i  # always >= 0, any integer (no parity constraint)
            acts = [STAY] * pad_steps
            cx, cy = x, y
            while cx != gx:
                if cx < gx:
                    acts.append(RIGHT); cx += 1
                else:
                    acts.append(LEFT); cx -= 1
            while cy != gy:
                if cy < gy:
                    acts.append(UP); cy += 1
                else:
                    acts.append(DOWN); cy -= 1
            plans.append(acts)

        assert all(len(p) == target_T for p in plans), "plan length should exactly equal target_T"

        ps0_cur = ps0
        realized_T = None
        realized_ter = None
        for t in range(target_T):
            pa0 = jnp.array([plans[i][t] for i in range(noa)])
            ps0_cur, _err, ter = env_step(ps0_cur, pa0, n, goal_set0)
            if int(ter) >= 1:
                realized_T = t + 1
                realized_ter = int(ter)
                break
        if realized_T is None:
            mismatches.append((combo, "never arrived", target_T))
            continue
        if realized_T != target_T:
            mismatches.append((combo, f"arrived at {realized_T}, expected {target_T}", None))
            continue
        if realized_ter != noa:
            mismatches.append((combo, f"ter={realized_ter}, expected full sync (noa={noa})", None))

    return mismatches


if __name__ == "__main__":
    n = 3
    goal_idx = n * n - 1
    best_rew = 10.0
    gamma = 0.9

    print("=== Verification against scripted simulation ===")
    for noa in (2, 3, 4):
        mismatches = verify_against_simulation(n, noa, goal_idx, best_rew, gamma, n_trials=300)
        print(f"noa={noa}: {len(mismatches)} mismatches out of 300 trials" + (f" -- {mismatches[:5]}" if mismatches else " (all match)"))

    print("\n=== Exact optimal expected reward (enumerated, no RL) ===")
    for noa in (2, 3, 4):
        opt = optimal_expected_reward(n, noa, goal_idx, best_rew, gamma)
        print(f"noa={noa}: optimal_expected_reward = {opt:.4f}")

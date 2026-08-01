"""Geometric-consensus rendezvous grid-world. Port of SAIC/envir_gc.m.

See PORT_NOTES.md SS2 for the full derivation and the deliberate deviation
from MATLAB (goal_set membership test, SS2 "Fixed, per explicit instruction").

Index mapping (MATLAB is 1-indexed; this module is 0-indexed throughout):

    MATLAB ps in {1..n^2}      <-> JAX ps0 in {0..n^2-1},  ps0 = ps - 1
    MATLAB pa in {1,2,3,4,5}   <-> JAX pa0 in {0,1,2,3,4}, pa0 = pa - 1
        pa0: 0=right, 1=left, 2=up, 3=down, 4=stay (same order as MATLAB)
    MATLAB goal_set (1-indexed cells) <-> JAX goal_set0 = goal_set - 1

MATLAB's cell numbering is column-major: ps = x + n*(y-1), where x is the
"right/left" coordinate and y is the "up/down" coordinate (both 1-indexed).
Converting to 0-indexed (x0 = x-1, y0 = y-1, ps0 = ps-1):

    ps0 = x0 + n*y0   =>   x0 = ps0 % n,  y0 = ps0 // n

`err` (per-agent illegal-move flag) is computed and returned for
completeness, matching MATLAB's output signature, but per PORT_NOTES.md SS2
every MATLAB caller discards it -- it carries no effect downstream.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

RIGHT, LEFT, UP, DOWN, STAY = 0, 1, 2, 3, 4


@partial(jax.jit, static_argnames=("n",))
def step(ps0: jnp.ndarray, pa0: jnp.ndarray, n: int, goal_set0: jnp.ndarray):
    """One environment transition for all agents at once.

    ps0: (noa,) int32, positions in [0, n^2).
    pa0: (noa,) int32, actions in {0..4} (RIGHT, LEFT, UP, DOWN, STAY).
    n: grid side length (static).
    goal_set0: (k,) int32, 0-indexed goal cells (k>=1).

    Returns (next_ps0, err, ter):
      next_ps0: (noa,) int32, updated positions.
      err: (noa,) bool, True where the chosen action was an illegal
           edge-crossing move (became a no-op) -- dead output, kept for
           parity with MATLAB's [ps,err,ter] signature.
      ter: int32 scalar, count of agents whose next_ps0 is a member of
           goal_set0 (PORT_NOTES.md SS2's membership-test fix: any(ps(i)
           == goal_set), not raw ps(i) == goal_set).
    """
    x0 = ps0 % n
    y0 = ps0 // n

    right_flag = x0 == (n - 1)
    left_flag = x0 == 0
    up_flag = y0 == (n - 1)
    down_flag = y0 == 0

    illegal_right = (pa0 == RIGHT) & right_flag
    illegal_left = (pa0 == LEFT) & left_flag
    illegal_up = (pa0 == UP) & up_flag
    illegal_down = (pa0 == DOWN) & down_flag
    err = illegal_right | illegal_left | illegal_up | illegal_down

    dx = jnp.where(
        (pa0 == RIGHT) & ~right_flag, 1, jnp.where((pa0 == LEFT) & ~left_flag, -1, 0)
    )
    dy = jnp.where((pa0 == UP) & ~up_flag, 1, jnp.where((pa0 == DOWN) & ~down_flag, -1, 0))

    next_x0 = x0 + dx
    next_y0 = y0 + dy
    next_ps0 = next_x0 + n * next_y0

    ter = jnp.sum(jnp.isin(next_ps0, goal_set0)).astype(jnp.int32)

    return next_ps0, err, ter


def at_goal(ps0: jnp.ndarray, goal_set0: jnp.ndarray) -> jnp.ndarray:
    """Per-agent membership test: is ps0[i] any of the goal cells?

    Shared helper for PORT_NOTES.md SS5.3's other two goal_set checks
    (action-canceling, rew_winner) so the same any(...) fix is applied
    consistently everywhere goal_set is compared against, not just here.
    """
    return jnp.isin(ps0, goal_set0)


def legit_states(n: int, goal_set0: jnp.ndarray) -> jnp.ndarray:
    """Non-goal cell list, precomputed once per (n, goal_set0) -- same
    Python-level construction centralized.py/train.py already do inline
    before their episode loops (goal_set0 is static per training run, so
    this is deliberately not jitted). Returns (n^2 - |goal_set0|,) int32.
    """
    n2 = n * n
    goal_set_py = set(int(g) for g in goal_set0.tolist())
    return jnp.array([s for s in range(n2) if s not in goal_set_py], dtype=jnp.int32)


@partial(jax.jit, static_argnames=("noa",))
def reset(rng: jax.Array, legit0: jnp.ndarray, noa: int) -> jnp.ndarray:
    """Generic episode init, factored out of centralized.py/train.py's
    existing inline logic (not wired into either -- purely additive, for
    esaic/certification.py's use): i.i.d. uniform draw over legit0, with
    replacement, independently per agent. Returns ps0: (noa,) int32.
    """
    n_legit = legit0.shape[0]
    idx = jax.random.randint(rng, (noa,), 0, n_legit)
    return legit0[idx]


@partial(jax.jit, static_argnames=("noa", "agent_idx"))
def reset_to(rng: jax.Array, legit0: jnp.ndarray, noa: int, agent_idx: int, o: jnp.ndarray) -> jnp.ndarray:
    """Like reset(), but agent `agent_idx` is pinned to observation `o`
    instead of drawn randomly -- every other agent still draws from the
    natural distribution. This is the "start an episode at observation o,
    marginalize over the other agent" convention already used by
    clustering.value_of_observation()/exact_value_of_observation.py,
    applied to a live rollout instead of exact enumeration. Used by
    esaic/certification.py's bracket-certificate MC rollouts.
    """
    ps0 = reset(rng, legit0, noa)
    return ps0.at[agent_idx].set(o)

"""Wraps jax_saic/env.py's rendezvous grid-world to satisfy
esaic.certification.EnvSpec -- the ONLY place rendezvous-specific code
meets the certification runtime path. Lives in validate/ (task-specific
glue, same role as this directory's other diagnostic scripts), not inside
esaic/ or jax_saic/, so neither of those ever imports the other.
"""
from __future__ import annotations

import pathlib
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from esaic.certification import EnvSpec
from jax_saic import env as rendezvous_env

NOA = 2  # matches jax_saic.centralized.NOA (ESAIC Theorem 1)


def make_rendezvous_env_spec(n: int, goal_set0: jnp.ndarray, best_rew: float) -> EnvSpec:
    """n: grid side length. goal_set0: 0-indexed goal cells. best_rew: full
    joint-arrival reward (matches CentralizedConfig.best_rew).

    step()'s reward is 0 at every non-terminal transition and best_rew (all
    NOA agents at goal) / 1.0 (some but not all) at the terminating step --
    this is the SAME reward jax_saic.centralized.train()'s per-episode
    summary reward implicitly encodes (best_rew * gamma**T / 1.0 * gamma**T
    for a T-step episode), just expressed per-step so a generic discounted
    MC rollout (sum_t gamma^t * r_t) reproduces it exactly, rather than a
    special-cased episode-summary formula.
    """
    n2 = n * n
    legit0 = rendezvous_env.legit_states(n, goal_set0)

    def step(joint_obs, joint_action, rng):
        del rng  # deterministic environment
        next_obs, _err, ter = rendezvous_env.step(joint_obs, joint_action, n, goal_set0)
        reward = jnp.where(ter == NOA, best_rew, jnp.where(ter >= 1, 1.0, 0.0))
        terminal = (ter >= 1).astype(jnp.int32)
        return next_obs, reward, terminal

    def reset(rng):
        return rendezvous_env.reset(rng, legit0, NOA)

    def reset_to(rng, agent_idx, o):
        return rendezvous_env.reset_to(rng, legit0, NOA, agent_idx, o)

    return EnvSpec(n_obs=n2, n_actions=5, n_agents=NOA, step=step, reset=reset, reset_to=reset_to)

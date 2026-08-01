"""Generic environment interface (G1) -- split into its own module so
certification.py and generic_centralized.py can both depend on it without
a circular import between them.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional


class EnvSpec(NamedTuple):
    """A plain bundle of callables/sizes, no behavior of its own -- callers
    (validate/, tests/) build one by wrapping a real environment (e.g.
    jax_saic.env for the rendezvous task) and pass it in; nothing under
    esaic/ imports a concrete environment directly (G1).

    n_obs: |Omega|, size of the single-agent observation space.
    n_actions: size of the single-agent action space A.
    n_agents: NOA for the centralized (joint) problem this EnvSpec drives
        (ESAIC Theorem 1: always 2, independent of the eventual
        decentralized agent count).
    step(joint_obs, joint_action, rng) -> (next_joint_obs, reward, terminal):
        one joint-MDP transition. joint_obs/joint_action: (n_agents,) int
        arrays. reward: scalar, the REPORTED reward (not a shaped training
        signal). terminal: bool/int scalar.
    reset(rng) -> joint_obs: draws a fresh joint episode-start state from
        the environment's own natural initial distribution.
    reset_to(rng, agent_idx, o) -> joint_obs: draws a joint episode-start
        state with agent `agent_idx` pinned to observation `o`, every other
        agent drawn from the natural distribution. Optional (None if
        unavailable) -- required only for cert_mode="bracket"'s
        V_pi_lower rollouts.
    """

    n_obs: int
    n_actions: int
    n_agents: int
    step: Callable
    reset: Callable
    reset_to: Optional[Callable] = None

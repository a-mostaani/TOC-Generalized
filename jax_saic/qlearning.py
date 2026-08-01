"""Decentralized position policy + update. Ports (PORT_NOTES.md SS6/SS7):

  ppolicy_customized_nbits_UCB_bestrew_ma.m  -> select_action()
  pupdate_customized_nbits_ma.m              -> update()

Both operate per-agent, given each agent's own (position, received-message)
pair, over the aggregated Q-table shared across all agents:

    qp_table shape: (noa, n^2, 2**((noa-1)*inf_bits), 5)
    NE_table shape: same (UCB visitation counts)

`decentralized_policy` (PORT_NOTES.md SS9 item 7, resolved) is a real,
user-selectable parameter here, independent of the centralized phase's own
`centralized_policy` (centralized.py) -- the two functions have different,
non-overlapping option sets and MATLAB hardcoded both to be unreachable in
different ways; neither hardcode is carried into this port.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

DECENTRALIZED_POLICIES = ("ep_greedy", "ucb", "greedy", "q_prob")

UCB_CONST_COEFF = 1.25 / 3  # ppolicy_...ma.m: ucb_const = 1.25*best_rew/3 (SS6.2;
# NOTE: different constant than centralized.py's 0.75/3 -- two independently
# tuned UCB schemes, do not conflate.


def _epsilon(counter, ns, end_learn):
    # SS6.1: epsilon = counter*(-0.98/(ns*end_learn)) + 1, floors at ~0.02 (no
    # explicit clamp), unlike centralized.py's ep-greedy which floors at 0.
    return counter * (-0.98 / (ns * end_learn)) + 1.0


def _ucb_bonus(qp_row, ne_row, best_rew, ucb_counter):
    ucb_const = UCB_CONST_COEFF * best_rew
    # SS6.2: log(ucb_counter+1) -- unlike centralized.py's log(ucb_counter)
    # without +1. Preserved exactly; the two phases' UCB formulas differ.
    return ucb_const * jnp.sqrt(jnp.log(ucb_counter + 1.0) / ne_row)


def select_action(
    ps0: jnp.ndarray,
    msg0: jnp.ndarray,
    qp_table: jnp.ndarray,
    NE_table: jnp.ndarray,
    *,
    policy: str,
    counter: int,
    ns: int,
    end_learn: float,
    best_rew: float,
    ucb_counter,
    tau: float,
    rng: jax.Array,
) -> jnp.ndarray:
    """Returns pa0: (noa,) int32, each agent's 0-indexed action.

    ps0: (noa,) each agent's own position, 0-indexed.
    msg0: (noa,) each agent's own received-message index, 0-indexed (see
          train.py for how this is built from the communicated bits).
    `policy` is a Python string (static, not traced) -- see module docstring.
    """
    noa = ps0.shape[0]
    qp_rows = qp_table[jnp.arange(noa), ps0, msg0]  # (noa, 5)

    if policy == "ep_greedy":
        return _act_ep_greedy(qp_rows, counter, ns, end_learn, rng)
    elif policy == "ucb":
        ne_rows = NE_table[jnp.arange(noa), ps0, msg0]
        return _act_ucb(qp_rows, ne_rows, best_rew, ucb_counter)
    elif policy == "greedy":
        return _act_greedy(qp_rows)
    elif policy == "q_prob":
        return _act_q_prob(qp_rows, tau, rng)
    else:
        raise ValueError(f"Unknown decentralized_policy {policy!r}; expected one of {DECENTRALIZED_POLICIES}")


@jax.jit
def _act_ep_greedy(qp_rows, counter, ns, end_learn, rng):
    noa = qp_rows.shape[0]  # static (shapes are always concrete under jit)
    epsilon = _epsilon(counter, ns, end_learn)
    rng_gate, rng_act = jax.random.split(rng)
    rr = jax.random.uniform(rng_gate, shape=(noa,))
    explore = rr <= epsilon  # MATLAB: rr <= epsilon (non-strict, unlike centralized's <)
    random_actions = jax.random.randint(rng_act, (noa,), 0, 5)
    greedy_actions = jnp.argmax(qp_rows, axis=-1)
    return jnp.where(explore, random_actions, greedy_actions).astype(jnp.int32)


@jax.jit
def _act_ucb(qp_rows, ne_rows, best_rew, ucb_counter):
    bonus = _ucb_bonus(qp_rows, ne_rows, best_rew, ucb_counter)
    return jnp.argmax(qp_rows + bonus, axis=-1).astype(jnp.int32)


@jax.jit
def _act_greedy(qp_rows):
    return jnp.argmax(qp_rows, axis=-1).astype(jnp.int32)


@jax.jit
def _act_q_prob(qp_rows, tau, rng):
    noa = qp_rows.shape[0]
    keys = jax.random.split(rng, noa)
    logits = qp_rows / tau
    return jax.vmap(jax.random.categorical)(keys, logits).astype(jnp.int32)


@jax.jit
def update(
    qp_table: jnp.ndarray,
    ps0: jnp.ndarray,
    last_ps0: jnp.ndarray,
    last_msg0: jnp.ndarray,
    msg0: jnp.ndarray,
    pa0: jnp.ndarray,
    temp_rew: float,
) -> jnp.ndarray:
    """Port of pupdate_customized_nbits_ma.m: SARSA-style, alpha=0.07,
    gamma=0.9 (hardcoded inside the MATLAB function, independent of any
    outer gamma -- SS7).

    Per SS7's resolved credit-assignment question: MATLAB's two nonzero-
    temp_rew branches ("all agents in rew_winner" vs. "partial arrival")
    compute the IDENTICAL update for every agent once temp_rew != 0 (the
    rew_winner-gated branch's own else-arm is dead code, since it only
    fires when literally every agent is already in rew_winner) -- so this
    collapses to a plain two-case update with no rew_winner dependence at
    all, matching what MATLAB actually computes, not just what its
    branching suggests.
    """
    noa = ps0.shape[0]
    alpha = 0.07
    gamma = 0.9
    agents = jnp.arange(noa)

    old = qp_table[agents, last_ps0, last_msg0, pa0]
    bootstrapped_target = gamma * jnp.max(qp_table[agents, ps0, msg0], axis=-1)
    target = jnp.where(temp_rew == 0.0, bootstrapped_target, temp_rew)
    new_val = old + alpha * (target - old)
    return qp_table.at[agents, last_ps0, last_msg0, pa0].set(new_val)

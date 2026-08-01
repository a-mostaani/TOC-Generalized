"""Decentralized training driver + Phase 1 orchestration.

Ports EoC_SAIC_3Agents.m (PORT_NOTES.md SS5), and ties together the three
phases of Algorithm 1 (PORT_NOTES.md SS4.4):

    centralized.py (noa=2, always)  ->  clustering.py  ->  this module (noa=N)

Index mapping: everything here is 0-indexed. `ag_states` (from
clustering.cluster_states) maps a raw position 0..n^2-1 to its 0-indexed
cluster id via s_aggregate() below -- the runtime realization of the
"learned communication policy" (PORT_NOTES.md SS0.2).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_saic import centralized, clustering, qlearning as ql
from jax_saic.env import STAY
from jax_saic.env import step as env_step
from jax_saic.indexing import flatten_mixed_radix, unflatten_mixed_radix

STUCK_LIMIT = 10  # SS5.3 step 8


class DecentralizedConfig(NamedTuple):
    n: int
    noa: int
    inf_bits: int  # bits carrying the aggregated cluster id (SS0.5: bits == inf_bits)
    goal_set0: jnp.ndarray
    best_rew: float
    gamma: float = 0.9  # only feeds tau's schedule (SS5.2); not threaded into qlearning.update
    tau_k: float = 0.005
    ns: int = 10_000  # MATLAB comment defaults: 10_000 (noa=2), 300_000 (noa=3), 1_500_000 (noa=4), all n=3
    end_learn: float = 0.80
    policy: str = "ep_greedy"
    update_tables: bool = True


def s_aggregate(ps0: jnp.ndarray, ag_states: jnp.ndarray) -> jnp.ndarray:
    """Look up each agent's cluster id in ag_states (clustering.cluster_states'
    output: (K, max_size) int32, 0-indexed raw states, -1 padding). This
    lookup IS the learned communication policy (PORT_NOTES.md SS0.2/SS4.3).
    """
    matches = ag_states[None, :, :] == ps0[:, None, None]  # (noa, K, max_size)
    in_cluster = jnp.any(matches, axis=-1)  # (noa, K)
    return jnp.argmax(in_cluster, axis=-1).astype(jnp.int32)


def _other_agent_index_table(noa: int) -> jnp.ndarray:
    """(noa, noa-1) static index table: row i lists every agent index except i,
    in ascending order -- used to build each agent's received-message vector
    from everyone else's outgoing message. noa is a plain Python int (static),
    built once per training run, not per step."""
    return jnp.array([[j for j in range(noa) if j != i] for i in range(noa)], dtype=jnp.int32)


def _tau_schedule(episode: int, tau_k: float) -> float:
    # SS5.2 step 1: tau = 1/(1+i*tau_k) for i<=40000, else frozen at i=40000's value.
    # Only read by decentralized_policy="q_prob" (SS6.1); inert otherwise.
    i = min(episode, 40_000)
    return 1.0 / (1.0 + i * tau_k)


def train(cfg: DecentralizedConfig, ag_states: jnp.ndarray, rng: jax.Array):
    """Runs the decentralized training loop (EoC_SAIC_3Agents.m, SS5) for
    `cfg.noa` agents, using a fixed `ag_states` aggregation table (produced
    once, at noa=2, by centralized.py + clustering.py -- SS0.8/ESAIC Theorem
    1). Returns `rew`: (ns,) float array, the per-episode reported reward
    (SS5.4 -- NOT the shaped temp_rew used for Q-updates, SS5.3.9), and the
    final qp_table.
    """
    noa = cfg.noa
    n2 = cfg.n * cfg.n
    n_msg = 2 ** ((noa - 1) * cfg.inf_bits)
    other_idx = _other_agent_index_table(noa)
    goal_set0 = cfg.goal_set0

    qp_table = jnp.full((noa, n2, n_msg, 5), 0.02)
    NE_table = jnp.full((noa, n2, n_msg, 5), 0.001)
    NE_table_emerged = jnp.full((noa, n2, n_msg, 5), 0.001)

    goal_set_py = set(int(g) for g in goal_set0.tolist())
    legit0 = jnp.array([s for s in range(n2) if s not in goal_set_py], dtype=jnp.int32)
    n_legit = legit0.shape[0]

    total_steps = 0
    rew = jnp.zeros((cfg.ns,))

    for episode in range(cfg.ns):
        rng, k_pos = jax.random.split(rng)
        tau = _tau_schedule(episode, cfg.tau_k)

        idx = jax.random.randint(k_pos, (noa,), 0, n_legit)
        ps0 = legit0[idx]
        ag_ps0 = s_aggregate(ps0, ag_states)

        stuck_counter = 0
        counter = 1  # MATLAB counter(i): 1-indexed step count within this episode
        temp_rew = 0.0
        term_this_episode = 0  # MATLAB term_hist(i): the ter value that ended this episode, 0 if stuck-terminated
        last_ps0 = None
        last_msg0 = None
        pa0 = None

        while True:
            # Encode: each agent's outgoing message is its own cluster id, in bits.
            ca0 = unflatten_mixed_radix(ag_ps0, base=2, k=cfg.inf_bits)  # (noa, inf_bits)
            # Rate-limited-but-PERFECT channel (SS0.5): every OTHER agent's
            # current message arrives verbatim; bsc_ch.m is not called.
            cs0 = ca0[other_idx].reshape(noa, -1)  # (noa, (noa-1)*inf_bits)
            msg0 = flatten_mixed_radix(cs0, base=2)  # (noa,)

            if counter != 1:
                qp_table = ql.update(qp_table, ps0, last_ps0, last_msg0, msg0, pa0, temp_rew)
                if term_this_episode >= 1:
                    break

            total_steps += 1
            rng, k_act = jax.random.split(rng)
            pa0 = ql.select_action(
                ps0,
                msg0,
                qp_table,
                NE_table,
                policy=cfg.policy,
                counter=episode,
                ns=cfg.ns,
                end_learn=cfg.end_learn,
                best_rew=cfg.best_rew,
                ucb_counter=float(total_steps),
                tau=tau,
                rng=k_act,
            )

            NE_table = NE_table.at[jnp.arange(noa), ps0, msg0, pa0].add(1)
            if episode > cfg.end_learn * cfg.ns:
                NE_table_emerged = NE_table_emerged.at[jnp.arange(noa), ps0, msg0, pa0].add(1)

            # Per-agent action-cancel at goal (SS5.3 step 6 -- unlike the
            # centralized phase's whole-team freeze, centralized.py point 1).
            at_goal = jnp.isin(ps0, goal_set0)
            pa0 = jnp.where(at_goal, STAY, pa0)

            last_ps0 = ps0
            last_msg0 = msg0
            ps0, _err, ter = env_step(ps0, pa0, cfg.n, goal_set0)
            ag_ps0 = s_aggregate(ps0, ag_states)

            all_unchanged = bool(jnp.all(last_ps0 == ps0))
            if all_unchanged:
                stuck_counter += 1
                if stuck_counter == STUCK_LIMIT:
                    temp_rew = 0.0
                    # "Unstick" hack (SS5.3 step 8): overwrite each agent's
                    # just-taken (state,action) value with the median of
                    # that state's whole action-row, discouraging repeating
                    # the same self-defeating action. Ported as-is, not
                    # "fixed" -- see PORT_NOTES.md SS5.3.
                    medians = jnp.median(
                        qp_table[jnp.arange(noa), last_ps0, last_msg0], axis=-1
                    )
                    qp_table = qp_table.at[jnp.arange(noa), last_ps0, last_msg0, pa0].set(medians)
                    term_this_episode = 0
                    # MATLAB's stuck-break (line 319) happens BEFORE
                    # counter(i)=counter(i)+1 (line 387) -- verified against
                    # source, not assumed. No effect on rew[] either way
                    # (the stuck/else branch below reports 0 regardless of
                    # counter's value), but matching exactly rather than
                    # relying on that being true.
                    break
            else:
                stuck_counter = 0

            ter_int = int(ter)  # one device sync, reused below
            if ter_int >= 1:
                # rew_winner length == ter always (both count "agents at any
                # goal cell", same membership test -- see train.py's design
                # notes / PORT_NOTES.md SS5.3.9). Shaped training signal, NOT
                # the reported reward (that's rew[episode] below, SS5.4).
                temp_rew = cfg.best_rew ** (ter_int - 1)
                term_this_episode = ter_int

            counter += 1

        # Episode summary reward (SS5.4) -- reported/plotted, distinct from
        # the shaped temp_rew that drove the Q-updates above.
        if term_this_episode == noa:
            ep_rew = cfg.best_rew * cfg.gamma ** (counter - 1)
        elif term_this_episode >= 1:
            ep_rew = 1.0 * cfg.gamma ** (counter - 1)
        else:
            ep_rew = 0.0
        rew = rew.at[episode].set(ep_rew)

    return rew, qp_table


def run_phase1(
    centralized_cfg: centralized.CentralizedConfig,
    decentralized_cfg: DecentralizedConfig,
    rng: jax.Array,
    clustering_method: str = "kmedian",
):
    """Full Algorithm 1 pipeline (PORT_NOTES.md SS4.4): centralized training
    (always noa=2) -> value-of-observation -> clustering -> decentralized
    training at decentralized_cfg.noa (any N). Returns (rew, ag_states,
    qp_table_decentralized).
    """
    rng_central, rng_decentral = jax.random.split(rng)

    qp_central, N_emerged_central, _central_rew = centralized.train(centralized_cfg, rng_central)
    V_o, N_o = clustering.value_of_observation(qp_central, N_emerged_central, centralized_cfg.n)
    ag_states_np = clustering.cluster_states(
        V_o, N_o, decentralized_cfg.inf_bits, method=clustering_method
    )
    ag_states = jnp.asarray(ag_states_np)

    rew, qp_decentral = train(decentralized_cfg, ag_states, rng_decentral)
    return rew, ag_states, qp_decentral

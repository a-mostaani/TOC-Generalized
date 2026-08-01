"""Centralized training phase. Ports (PORT_NOTES.md SS4.1):

  Fully Centralized - MultiAgent/benchmark_perfectcom_MultiAgent.m (driver)
  Fully Centralized - MultiAgent/bench_policy_UCB.m               (centralized_policy)
  Fully Centralized - MultiAgent/pbench_update.m                  (Q-update)

Per PORT_NOTES.md SS0.8 (ESAIC Theorem 1), this ALWAYS runs at noa=2,
regardless of the decentralized phase's target agent count -- the whole
point of ESAIC is that this expensive phase doesn't need to scale with N.
A single run is performed (PORT_NOTES.md SS9 item 9, resolved), not
MATLAB's bn=5 batches.

Index mapping: joint state/action are flattened 0-indexed scalars via
indexing.flatten_mixed_radix / unflatten_mixed_radix (see that module's
docstring for the derivation from MATLAB's ps_calc/pa_calc/mps_calc):

    main_ps0 in [0, (n^2)^2)  <-> MATLAB main_ps in [1, (n^2)^2]
    main_pa0 in [0, 5^2)      <-> MATLAB main_pa in [1, 5^2]

Two behaviors found only while implementing this (not in the original
PORT_NOTES.md draft -- flagging here, not silently absorbing them):

  1. Action-cancellation freezes the WHOLE joint action, not just the
     arrived agent: `if sum(ps==goal_set)>=1: main_pa=5^noa` (line 235-236)
     forces ALL agents to STAY the instant ANY one agent reaches a goal
     cell -- unlike the decentralized phase (SS5.3 step 6), which only
     freezes the individual agent(s) at goal. Ported faithfully.
  2. Two separate Q-update call sites exist, not one: an unconditional one
     at the top of the loop (updates the *previous* iteration's transition,
     using that iteration's resulting temp_rew) and a second one only at
     loop-break time, gated by `i < end_learn*ns` (updates the *current*,
     terminal transition -- the one that actually carries the positive
     reward). Past `end_learn*ns`, intermediate zero-reward TD updates
     keep happening every step, but the terminal positive-reward update is
     skipped. This looks like an inconsistency (a true "stop learning"
     freeze would gate both sites), but it's exactly what the MATLAB does;
     ported verbatim, not fixed.
  3. `rew_winner` is computed in MATLAB but never read anywhere in this
     file (temp_rew depends only on `ter`, not on rew_winner's length --
     unlike the decentralized phase). Dropped from the port entirely, same
     treatment as `worst_rew` (PORT_NOTES.md SS8): confirmed dead, not
     carried forward even as inert plumbing.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_saic.env import step as env_step
from jax_saic.indexing import flatten_mixed_radix, unflatten_mixed_radix

NOA = 2  # centralized training always runs at noa=2 (PORT_NOTES.md SS0.8)
ALPHA = 0.07
GAMMA = 0.9  # hardcoded inside pbench_update.m, independent of any outer gamma (SS4.1)
UCB_CONST_COEFF = 0.75 / 3  # bench_policy_UCB.m: ucb_const = 0.75*best_rew/3

CENTRALIZED_POLICIES = (
    "ep-greedy",
    "ucb",
    "stochastic-ucb",
    "stochastic-epsilon",
    "stochastic",
)


class CentralizedConfig(NamedTuple):
    n: int
    goal_set0: jnp.ndarray  # 0-indexed goal cells
    best_rew: float
    ns: int  # centralized episodes; MATLAB comment default for noa=2,n=3 is 120_000
    end_learn: float  # fraction of ns; MATLAB default 0.850 (distinct from decentralized's 0.80)
    policy: str = "ep-greedy"
    tau: float = 0.005  # only read by stochastic-ucb/stochastic-epsilon/stochastic
    use_double_q: bool = False  # esaic/certification.py: van Hasselt double Q-learning,
    # additive/opt-in -- False reproduces the original single-Q path exactly (see train()).


def _epsilon(episode, ns, end_learn):
    # bench_policy_UCB.m: epsilon = init_epsilon - episode/(end_learn*ns), init_epsilon=1
    # Anneals linearly to exactly 0 at episode=end_learn*ns, unclamped beyond (SS4.1).
    return 1.0 - episode / (end_learn * ns)


def _ucb_bonus(qp_row, N_row, best_rew, ucb_counter):
    ucb_const = UCB_CONST_COEFF * best_rew
    return ucb_const * jnp.sqrt(jnp.log(ucb_counter) / N_row)


def select_action(
    main_ps0: jnp.ndarray,
    qp_table: jnp.ndarray,
    N_table: jnp.ndarray,
    *,
    policy: str,
    episode: int,
    ns: int,
    end_learn: float,
    best_rew: float,
    ucb_counter,
    tau: float,
    noa: int,
    rng: jax.Array,
) -> jnp.ndarray:
    """Returns main_pa0 (0-indexed flattened joint action, scalar int32).

    `policy` is a Python string, resolved here (not traced) -- each policy
    is its own small jitted function so JAX only ever compiles the branch
    actually used for a given training run.
    """
    n_actions = 5**noa
    qp_row = qp_table[main_ps0]
    update_window = episode < ns * end_learn

    if policy == "ep-greedy":
        return _act_ep_greedy(qp_row, episode, ns, end_learn, n_actions, rng)
    elif policy == "ucb":
        return _act_ucb(qp_row, N_table[main_ps0], best_rew, ucb_counter, update_window)
    elif policy == "stochastic-ucb":
        return _act_stochastic_ucb(
            qp_row, N_table[main_ps0], best_rew, ucb_counter, tau, update_window, rng
        )
    elif policy == "stochastic-epsilon":
        return _act_stochastic_epsilon(qp_row, episode, ns, end_learn, tau, update_window, rng)
    elif policy == "stochastic":
        return _act_stochastic(qp_row, tau, rng)
    else:
        raise ValueError(f"Unknown centralized_policy {policy!r}; expected one of {CENTRALIZED_POLICIES}")


@jax.jit
def _act_ep_greedy(qp_row, episode, ns, end_learn, n_actions, rng):
    epsilon = _epsilon(episode, ns, end_learn)
    rng_gate, rng_act = jax.random.split(rng)
    explore = jax.random.uniform(rng_gate) < epsilon  # MATLAB: ran < epsilon (strict)
    random_action = jax.random.randint(rng_act, (), 0, n_actions)
    greedy_action = jnp.argmax(qp_row)
    return jnp.where(explore, random_action, greedy_action).astype(jnp.int32)


@jax.jit
def _act_ucb(qp_row, N_row, best_rew, ucb_counter, update_window):
    bonus = _ucb_bonus(qp_row, N_row, best_rew, ucb_counter)
    ucb_action = jnp.argmax(qp_row + bonus)
    greedy_action = jnp.argmax(qp_row)
    return jnp.where(update_window, ucb_action, greedy_action).astype(jnp.int32)


@jax.jit
def _act_stochastic_ucb(qp_row, N_row, best_rew, ucb_counter, tau, update_window, rng):
    bonus = _ucb_bonus(qp_row, N_row, best_rew, ucb_counter)
    logits = (qp_row + bonus) / tau
    sampled = jax.random.categorical(rng, logits)
    greedy_action = jnp.argmax(qp_row)
    return jnp.where(update_window, sampled, greedy_action).astype(jnp.int32)


@jax.jit
def _act_stochastic_epsilon(qp_row, episode, ns, end_learn, tau, update_window, rng):
    epsilon = _epsilon(episode, ns, end_learn)
    rng_gate, rng_act = jax.random.split(rng)
    explore = jax.random.uniform(rng_gate) < epsilon
    logits = qp_row / tau
    sampled = jax.random.categorical(rng_act, logits)
    greedy_action = jnp.argmax(qp_row)
    chosen = jnp.where(explore, sampled, greedy_action)
    return jnp.where(update_window, chosen, greedy_action).astype(jnp.int32)


@jax.jit
def _act_stochastic(qp_row, tau, rng):
    # MATLAB's 'stochastic' branch has no update-window gate at all (SS4.1) --
    # always samples stochastically, unlike the other four branches.
    logits = qp_row / tau
    return jax.random.categorical(rng, logits).astype(jnp.int32)


@jax.jit
def update(qp_table, main_ps0, main_last_ps0, main_pa0, temp_rew):
    """Port of pbench_update.m: off-policy (max-bootstrap) Q-learning."""
    old = qp_table[main_last_ps0, main_pa0]
    bootstrapped_target = GAMMA * jnp.max(qp_table[main_ps0])
    target = jnp.where(temp_rew == 0.0, bootstrapped_target, temp_rew)
    new_val = old + ALPHA * (target - old)
    return qp_table.at[main_last_ps0, main_pa0].set(new_val)


def _double_update_one(q_update, q_other, main_ps0, main_last_ps0, main_pa0, temp_rew):
    old = q_update[main_last_ps0, main_pa0]
    a_star = jnp.argmax(q_update[main_ps0])
    bootstrapped_target = GAMMA * q_other[main_ps0, a_star]
    target = jnp.where(temp_rew == 0.0, bootstrapped_target, temp_rew)
    new_val = old + ALPHA * (target - old)
    return q_update.at[main_last_ps0, main_pa0].set(new_val)


@jax.jit
def double_update(Q_A, Q_B, main_ps0, main_last_ps0, main_pa0, temp_rew, rng):
    """esaic/certification.py: standard tabular Double Q-learning (van Hasselt
    2010, Algorithm 1; van Hasselt, Guez & Silver 2016). Each call, with
    probability 0.5, update ONE of the two tables: the table being updated
    supplies the greedy action (argmax) at the new state, but the OTHER
    table supplies that action's bootstrapped value. Decoupling action
    selection from value evaluation this way removes the single-Q update's
    systematic overestimation bias. Branches on a PRNG bit via jax.lax.cond
    (not Python `if`) so this stays jit-compatible.
    """
    update_a = jax.random.bernoulli(rng)

    def branch_a(qa, qb):
        return _double_update_one(qa, qb, main_ps0, main_last_ps0, main_pa0, temp_rew), qb

    def branch_b(qa, qb):
        return qa, _double_update_one(qb, qa, main_ps0, main_last_ps0, main_pa0, temp_rew)

    return jax.lax.cond(update_a, branch_a, branch_b, Q_A, Q_B)


def train(
    cfg: CentralizedConfig,
    rng: jax.Array,
    focus_states: jnp.ndarray | None = None,
    focus_frac: float = 0.0,
):
    """Runs the full centralized-training driver (benchmark_perfectcom_MultiAgent.m,
    at noa=2, a single run -- PORT_NOTES.md SS4.4). Returns (qp_table,
    N_table_emerged, rew): the first two shape ((n^2)^2, 5^2); rew is
    shape (cfg.ns,), the per-episode reported reward (PORT_NOTES.md
    SS11.3), matching the reference script's own rew(i) tracking.

    esaic/certification.py additions, both additive/opt-in, default OFF:
      - cfg.use_double_q=True: maintains two Q-tables (double_update()
        above) instead of one; returns 0.5*(Q_A+Q_B) as qp_table.
      - focus_states/focus_frac: when focus_states is not None, agent
        index 1's (the same index clustering.value_of_observation()
        computes V_o for) episode-start position is drawn from
        focus_states with probability focus_frac, else from the natural
        distribution as usual; agent 0 always draws naturally. This is
        the refinement loop's "oversample near-boundary observations"
        hook. Both default to a no-op that reproduces the pre-existing
        code path byte-for-byte (verified: no extra PRNG draw happens
        when focus_states is None, so the RNG stream is unperturbed).
    """
    n2 = cfg.n * cfg.n
    n_states = n2**NOA
    n_actions = 5**NOA
    goal_set0 = cfg.goal_set0

    if cfg.use_double_q:
        Q_A = jnp.full((n_states, n_actions), 0.02)
        Q_B = jnp.full((n_states, n_actions), 0.02)
    else:
        qp_table = jnp.full((n_states, n_actions), 0.02)
    N_table = jnp.full((n_states, n_actions), 0.001)
    N_table_emerged = jnp.full((n_states, n_actions), 0.001)

    goal_set_py = set(int(g) for g in goal_set0.tolist())
    legit0 = jnp.array([s for s in range(n2) if s not in goal_set_py], dtype=jnp.int32)
    n_legit = legit0.shape[0]

    total_steps = 0  # ucb_counter = sum(counter(1:i)) across all episodes so far
    rew = jnp.zeros((cfg.ns,))  # matches benchmark_perfectcom_MultiAgent.m's own rew(i) tracking
    # (this port's train() previously only returned qp_table/N_table_emerged --
    # the reward curve itself wasn't tracked, since the pipeline only needed
    # the Q-table for value-of-observation/clustering. Added to answer "is
    # the centralized phase itself converging well" directly, not just assumed.)

    for episode in range(cfg.ns):
        rng, k_init_pos, k_init_act = jax.random.split(rng, 3)

        # ps_ind=randi(n^2-1,noa,1); ps(kk)=s_space(ps_ind(kk))  -- i.i.d. draws
        # WITH replacement from the non-goal cells, independently per agent.
        if focus_states is None:
            idx = jax.random.randint(k_init_pos, (NOA,), 0, n_legit)
            ps0 = legit0[idx]
        else:
            # Agent 0 always natural; agent 1 (clustering.value_of_observation's
            # marginalized index) mixes in focus_states with prob focus_frac.
            # Only entered when focus_states is set, so the branch above is
            # byte-for-byte unperturbed when this feature is unused.
            k_a0, k_a1_nat, k_a1_focus, k_a1_gate = jax.random.split(k_init_pos, 4)
            agent0_pos = legit0[jax.random.randint(k_a0, (), 0, n_legit)]
            natural_agent1 = legit0[jax.random.randint(k_a1_nat, (), 0, n_legit)]
            focus_agent1 = focus_states[jax.random.randint(k_a1_focus, (), 0, focus_states.shape[0])]
            use_focus = jax.random.uniform(k_a1_gate) < focus_frac
            agent1_pos = jnp.where(use_focus, focus_agent1, natural_agent1)
            ps0 = jnp.stack([agent0_pos, agent1_pos])
        main_ps0 = flatten_mixed_radix(ps0, base=n2)
        # main_pa=randi(5^noa) initial draw is immediately overwritten inside
        # the loop before use -- wasted entropy, matching MATLAB (not ported,
        # since it has zero effect on any output; see PORT_NOTES.md SS5.2
        # for the decentralized phase's analogous wasted draws, kept there
        # only for documentation completeness).

        temp_rew = 0.0
        step_in_episode = 1  # MATLAB counter(i), 1-indexed step count within this episode
        last_main_ps0 = None
        last_main_pa0 = None

        while True:
            if step_in_episode != 1:
                if cfg.use_double_q:
                    rng, k_dq = jax.random.split(rng)
                    Q_A, Q_B = double_update(
                        Q_A, Q_B, main_ps0, last_main_ps0, last_main_pa0, temp_rew, k_dq
                    )
                else:
                    qp_table = update(qp_table, main_ps0, last_main_ps0, last_main_pa0, temp_rew)

            total_steps += 1
            ucb_counter = float(total_steps)
            rng, k_act = jax.random.split(rng)
            main_pa0 = select_action(
                main_ps0,
                0.5 * (Q_A + Q_B) if cfg.use_double_q else qp_table,
                N_table,
                policy=cfg.policy,
                episode=episode,
                ns=cfg.ns,
                end_learn=cfg.end_learn,
                best_rew=cfg.best_rew,
                ucb_counter=ucb_counter,
                tau=cfg.tau,
                noa=NOA,
                rng=k_act,
            )

            N_table = N_table.at[main_ps0, main_pa0].add(1)
            if episode > cfg.end_learn * cfg.ns:
                N_table_emerged = N_table_emerged.at[main_ps0, main_pa0].add(1)

            # Freeze the WHOLE joint action if ANY agent is currently at a
            # goal cell (see module docstring point 1 -- not per-agent like
            # the decentralized phase).
            any_at_goal = jnp.any(jnp.isin(ps0, goal_set0))
            all_stay0 = n_actions - 1
            main_pa0 = jnp.where(any_at_goal, all_stay0, main_pa0)

            last_main_ps0 = main_ps0
            last_main_pa0 = main_pa0
            pa0 = unflatten_mixed_radix(main_pa0, base=5, k=NOA)
            ps0, _err, ter = env_step(ps0, pa0, cfg.n, goal_set0)
            main_ps0 = flatten_mixed_radix(ps0, base=n2)

            if int(ter) == NOA:
                temp_rew = float(cfg.best_rew)
            elif int(ter) >= 1:
                temp_rew = 1.0

            step_in_episode += 1

            if int(ter) >= 1:
                if episode < cfg.end_learn * cfg.ns:
                    if cfg.use_double_q:
                        rng, k_dq = jax.random.split(rng)
                        Q_A, Q_B = double_update(
                            Q_A, Q_B, main_ps0, last_main_ps0, last_main_pa0, temp_rew, k_dq
                        )
                    else:
                        qp_table = update(qp_table, main_ps0, last_main_ps0, last_main_pa0, temp_rew)
                break

        # Episode summary reward (benchmark_perfectcom_MultiAgent.m: "if
        # temp_rew>1 ... else ..."). No stuck-escape branch exists in the
        # centralized script (PORT_NOTES.md SS4, confirmed via source grep),
        # so unlike the decentralized phase's rew(), this is never 0.
        if temp_rew > 1:
            ep_rew = cfg.best_rew * (GAMMA ** (step_in_episode - 1))
        else:
            ep_rew = 1.0 * (GAMMA ** (step_in_episode - 1))
        rew = rew.at[episode].set(ep_rew)

    if cfg.use_double_q:
        qp_table = 0.5 * (Q_A + Q_B)
    return qp_table, N_table_emerged, rew

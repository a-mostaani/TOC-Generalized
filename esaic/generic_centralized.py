"""Generic tabular centralized (NOA=2) Q-learning trainer, driven entirely
through esaic.certification.EnvSpec.

Why this exists instead of reusing jax_saic.centralized.train(): that
function is hardcoded to `from jax_saic.env import step as env_step` (fine
for Phase 1's single-task scope), so it cannot run on any environment but
the rendezvous grid -- reusing it here would make the certification/
refinement runtime path implicitly rendezvous-only, failing G1's
cross-task-generalization requirement (spec S6 test 7). It also isn't
safely reusable piecemeal: jax_saic.centralized.select_action() computes
`n_actions = 5**noa` (hardcoded per-agent action count) rather than reading
it from the table/env, and update()/double_update() close over module-level
GAMMA=0.9/ALPHA=0.07 constants rather than taking them as parameters -- both
are exactly the kind of hidden task-specific assumption G1 forbids. So this
module reimplements the same *algorithm* (off-policy tabular Q-learning,
optional double-Q, epsilon-greedy, optional focus-biased exploring starts)
fully parameterized by env.n_obs/env.n_actions/gamma/alpha instead.

Does NOT replicate jax_saic.centralized.train()'s MATLAB-fidelity quirks
(whole-joint-action freeze at goal, stuck-counter escape) -- those are
documented Phase-1 rendezvous-fidelity choices (PORT_NOTES.md SS4.1), not
general ESAIC algorithm requirements; this trainer relies only on
EnvSpec.step's own terminal signal.

Array sizes: qp_table/N_table are [n_obs**n_agents, n_actions**n_agents] --
fixed by n_agents (NOA=2 for ESAIC's centralized phase), never scaling with
the decentralized target N (G2).
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from esaic.env_spec import EnvSpec
from jax_saic.indexing import flatten_mixed_radix, unflatten_mixed_radix


class GenericCentralizedConfig(NamedTuple):
    ns: int
    end_learn: float
    gamma: float
    alpha: float = 0.07
    epsilon_start: float = 1.0
    use_double_q: bool = False


def _epsilon(episode, ns, end_learn, epsilon_start):
    return epsilon_start * (1.0 - episode / (end_learn * ns))


def _random_tiebreak_argmax(rng, values):
    """argmax with ties broken uniformly at random, instead of jnp.argmax's
    implicit "lowest index wins". Found to matter directly (not a
    theoretical concern): every (state, joint-action) pair starts EXACTLY
    tied at the shared 0.02 init, and jnp.argmax's tie-break systematically
    preferred joint-action index 0 (= "both agents move RIGHT", under the
    agent-0-fastest flattening -- indexing.py) whenever a greedy step hit
    still-tied Q-values, i.e. constantly early in training and for
    infrequently-updated (state, action) pairs throughout. Measured
    consequence in a real run: the x=0 grid column (farthest from a
    goal at x=n-1) got 5-10x fewer visits than cells near the goal, and
    per-state value error vs. exact DP correlated -0.55 with visitation
    count -- i.e. this tie-break bias was silently degrading value
    estimates specifically for the states it steered exploration away
    from, not a cosmetic-only effect.

    Implementation: assign iid uniform random priority to every index
    tied for the max, -inf to every non-tied index, and take argmax of
    that -- standard random-tiebreak-argmax. A non-maximal value can never
    win (its priority is -inf); ties are broken uniformly since the
    priorities among them are iid.
    """
    is_max = values == jnp.max(values)
    priority = jnp.where(is_max, jax.random.uniform(rng, values.shape), -jnp.inf)
    return jnp.argmax(priority)


@jax.jit
def _select_action(qp_row, episode, ns, end_learn, epsilon_start, n_actions, rng):
    epsilon = jnp.clip(_epsilon(episode, ns, end_learn, epsilon_start), 0.0, 1.0)
    rng_gate, rng_act, rng_tie = jax.random.split(rng, 3)
    explore = jax.random.uniform(rng_gate) < epsilon
    random_action = jax.random.randint(rng_act, (), 0, n_actions)
    greedy_action = _random_tiebreak_argmax(rng_tie, qp_row)
    return jnp.where(explore, random_action, greedy_action).astype(jnp.int32)


@jax.jit
def _q_update(qp_table, main_ps0, main_last_ps0, main_pa0, temp_rew, gamma, alpha):
    old = qp_table[main_last_ps0, main_pa0]
    bootstrapped_target = gamma * jnp.max(qp_table[main_ps0])
    target = jnp.where(temp_rew == 0.0, bootstrapped_target, temp_rew)
    new_val = old + alpha * (target - old)
    return qp_table.at[main_last_ps0, main_pa0].set(new_val)


def _double_q_update_one(q_update, q_other, main_ps0, main_last_ps0, main_pa0, temp_rew, gamma, alpha):
    old = q_update[main_last_ps0, main_pa0]
    a_star = jnp.argmax(q_update[main_ps0])
    bootstrapped_target = gamma * q_other[main_ps0, a_star]
    target = jnp.where(temp_rew == 0.0, bootstrapped_target, temp_rew)
    new_val = old + alpha * (target - old)
    return q_update.at[main_last_ps0, main_pa0].set(new_val)


@jax.jit
def _double_q_update(Q_A, Q_B, main_ps0, main_last_ps0, main_pa0, temp_rew, gamma, alpha, rng):
    """Same van Hasselt double Q-learning as jax_saic.centralized.double_update
    (see that function's docstring), reimplemented here fully parameterized
    by gamma/alpha instead of closing over module constants."""
    update_a = jax.random.bernoulli(rng)

    def branch_a(qa, qb):
        return _double_q_update_one(qa, qb, main_ps0, main_last_ps0, main_pa0, temp_rew, gamma, alpha), qb

    def branch_b(qa, qb):
        return qa, _double_q_update_one(qb, qa, main_ps0, main_last_ps0, main_pa0, temp_rew, gamma, alpha)

    return jax.lax.cond(update_a, branch_a, branch_b, Q_A, Q_B)


def train(
    cfg: GenericCentralizedConfig,
    env: EnvSpec,
    rng: jax.Array,
    focus_states: Optional[jnp.ndarray] = None,
    focus_frac: float = 0.0,
    init_qp_table: Optional[jnp.ndarray] = None,
    init_N_table_emerged: Optional[jnp.ndarray] = None,
    track_reward: bool = False,
):
    """Returns (qp_table, N_table_emerged, total_steps): joint-indexed
    [n_obs**n_agents, n_actions**n_agents] arrays, matching
    jax_saic.centralized.train()'s return-shape convention (minus the
    per-episode `rew` curve, not needed by any certification caller).

    track_reward: optional, default False -- return signature is UNCHANGED
    unless this is explicitly set (every existing caller, including
    esaic.certification.run_esaic_certified's train_fn calls, leaves it at
    the default and is unaffected). When True, returns a 5-tuple instead:
    (qp_table, N_table_emerged, total_steps, rew, steps_at_episode_end).
    rew[i] is episode i's terminal reward (same convention as
    jax_saic.centralized.train()'s own `rew` curve); steps_at_episode_end[i]
    is this call's own total_steps counter at the moment episode i ended
    (i.e. 1-indexed within THIS call only, not across chained calls -- a
    caller chaining multiple train() calls, e.g. certification's refinement
    rounds via a custom train_fn, must add its own running step offset to
    get one continuous step axis; see validate/ for an example).

    focus_states/focus_frac: same "agent index 1 biased toward focus_states
    with probability focus_frac, agent 0 always natural" convention as
    jax_saic.centralized.train()'s additive hook -- see that function's
    docstring for why agent index 1 specifically (matches
    clustering.value_of_observation's marginalized index).

    init_qp_table/init_N_table_emerged: optional warm start, both None by
    default (matches the original fresh-init behavior exactly). Without
    this, esaic.certification.run_esaic_certified()'s refinement loop was
    calling this function fresh every round -- each "refined" table was
    actually a from-scratch run of only `refine_steps` episodes, discarding
    all prior training (including the much larger initial run) rather than
    building on it, which is why certification was never actually reachable
    through refinement (found directly: state values after several rounds
    were LESS converged than after the initial run alone). When
    init_qp_table is given (run_esaic_certified now always passes the
    previous round's table forward), training continues from it instead of
    re-initializing at 0.02 everywhere. For use_double_q=True, both Q_A and
    Q_B are seeded from the same init_qp_table (the combined table is what
    callers have -- Q_A/Q_B aren't separately exposed); this only reduces
    double-Q's decorrelation benefit for the first few post-warm-start
    updates, since which table updates each step is still an independent
    random choice and they diverge again quickly.
    """
    n_states = env.n_obs**env.n_agents
    n_actions_joint = env.n_actions**env.n_agents

    if cfg.use_double_q:
        Q_A = init_qp_table if init_qp_table is not None else jnp.full((n_states, n_actions_joint), 0.02)
        Q_B = init_qp_table if init_qp_table is not None else jnp.full((n_states, n_actions_joint), 0.02)
    else:
        qp_table = init_qp_table if init_qp_table is not None else jnp.full((n_states, n_actions_joint), 0.02)
    N_table_emerged = (
        init_N_table_emerged if init_N_table_emerged is not None else jnp.full((n_states, n_actions_joint), 0.001)
    )
    total_steps = 0
    rew_list = [] if track_reward else None
    steps_list = [] if track_reward else None

    for episode in range(cfg.ns):
        rng, k_reset, k_gate, k_focus_pick = jax.random.split(rng, 4)

        if focus_states is None:
            joint_obs = env.reset(k_reset)
        else:
            use_focus = jax.random.uniform(k_gate) < focus_frac
            natural = env.reset(k_reset)
            focus_idx = jax.random.randint(k_focus_pick, (), 0, focus_states.shape[0])
            forced = env.reset_to(k_reset, 1, focus_states[focus_idx])
            joint_obs = jnp.where(use_focus, forced, natural)

        main_ps0 = flatten_mixed_radix(joint_obs, base=env.n_obs)

        temp_rew = 0.0
        step_in_episode = 1
        last_main_ps0 = None
        last_main_pa0 = None

        while True:
            if step_in_episode != 1:
                if cfg.use_double_q:
                    rng, k_dq = jax.random.split(rng)
                    Q_A, Q_B = _double_q_update(
                        Q_A, Q_B, main_ps0, last_main_ps0, last_main_pa0, temp_rew, cfg.gamma, cfg.alpha, k_dq
                    )
                else:
                    qp_table = _q_update(
                        qp_table, main_ps0, last_main_ps0, last_main_pa0, temp_rew, cfg.gamma, cfg.alpha
                    )

            total_steps += 1
            rng, k_act, k_step = jax.random.split(rng, 3)
            current_table = 0.5 * (Q_A + Q_B) if cfg.use_double_q else qp_table
            main_pa0 = _select_action(
                current_table[main_ps0], episode, cfg.ns, cfg.end_learn, cfg.epsilon_start, n_actions_joint, k_act
            )
            if episode > cfg.end_learn * cfg.ns:
                N_table_emerged = N_table_emerged.at[main_ps0, main_pa0].add(1)

            last_main_ps0 = main_ps0
            last_main_pa0 = main_pa0
            joint_action = unflatten_mixed_radix(main_pa0, base=env.n_actions, k=env.n_agents)
            joint_obs, reward, terminal = env.step(joint_obs, joint_action, k_step)
            main_ps0 = flatten_mixed_radix(joint_obs, base=env.n_obs)

            ter_int = int(terminal)
            if ter_int >= 1:
                temp_rew = float(reward)
            step_in_episode += 1
            if ter_int >= 1:
                if episode < cfg.end_learn * cfg.ns:
                    if cfg.use_double_q:
                        rng, k_dq = jax.random.split(rng)
                        Q_A, Q_B = _double_q_update(
                            Q_A, Q_B, main_ps0, last_main_ps0, last_main_pa0, temp_rew, cfg.gamma, cfg.alpha, k_dq
                        )
                    else:
                        qp_table = _q_update(
                            qp_table, main_ps0, last_main_ps0, last_main_pa0, temp_rew, cfg.gamma, cfg.alpha
                        )
                if track_reward:
                    rew_list.append(temp_rew)
                    steps_list.append(total_steps)
                break

    if cfg.use_double_q:
        qp_table = 0.5 * (Q_A + Q_B)
    if track_reward:
        return (
            qp_table,
            N_table_emerged,
            total_steps,
            jnp.asarray(rew_list),
            jnp.asarray(steps_list, dtype=jnp.int32),
        )
    return qp_table, N_table_emerged, total_steps

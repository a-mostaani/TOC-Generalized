"""Boundary-gate value certification + refinement for ESAIC's centralized
training phase. See CERTIFICATION.md for the math and citations.

WHY: ESAIC's centralized phase (jax_saic.centralized.train, NOA=2) produces
a value estimate V_hat over the single-agent observation space Omega
(via jax_saic.clustering.value_of_observation's marginalization) that is
only accurate in expectation under the centralized policy's own visitation
-- not uniformly over Omega. The state-aggregation (k-median) step that
turns V_hat into a communication codeword needs accuracy specifically NEAR
CLUSTER BOUNDARIES, for every observation any decentralized control policy
might later visit, or codewords get mis-assigned. This module adds: a
per-observation CERTIFICATE (model-free, sample-based) that upper-bounds
|V_hat(o) - V*(o)|, a GATE that compares the certificate to the observation's
distance from its nearest cluster boundary, and a REFINEMENT loop that
spends extra centralized-training sampling only on gate-failing (uncertified)
observations.

HARD GUARDRAILS (see the instruction this module implements; summarized here
so violations are visible on read, not just in a separate doc):

G1 -- No smuggling toy-example information into this module. Every function
below depends ONLY on: a value estimate over Omega, medoids/partition from
clustering, visitation counts, gamma, Rmax, and the generic EnvSpec interface
defined here. Nothing in this file imports jax_saic.env, references grid
geometry, a goal point, the reward constants C1/C2, the action set
{Right,Left,Up,Down,Stay}, an assumed value-function shape, or an exact-V*
oracle. The rendezvous adapter that satisfies EnvSpec lives in jax_saic/env.py
and is constructed by CALLERS (validate/, tests/), never imported here.

G2 -- No solutions that only work at toy scale. Every array here is indexed
by observation o in Omega (shape [|Omega|] or [|Omega|, B]) or by action
(+ [A] axis), NEVER by joint state s in Omega^N -- nothing scales with N or
|Omega|^N. No full-sweep value iteration / exact planning / joint-space
enumeration appears on this runtime path (those exist only as test oracles,
under tests/). Target complexity, honored below: gate evaluation O(|Omega|*B)
time/memory; certificate cost independent of N; refinement cost bounded by
CertificationConfig.sampling_budget. Vectorized over Omega; no Python-level
loop over individual observations.

Function-approximation caveat (spec S7): the ell-infinity guarantee here is
honest only in the tabular regime with a controllable exploration floor. If
V_hat ever comes from a neural network instead of a table, U(o) from
ensembles/counts becomes a proxy, not a bound -- off-distribution
extrapolation has no sup-norm guarantee. The bracket form's MC lower bound
(V_pi_lower) stays valid there; the optimistic upper bound (Q_ub) does not.
"""
from __future__ import annotations

import warnings
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from esaic.env_spec import EnvSpec
from jax_saic.clustering import cluster_states
from jax_saic.indexing import flatten_mixed_radix, unflatten_mixed_radix

# indexing.py is a generic 0-indexed mixed-radix (de)flatten utility -- joint
# state/action <-> per-agent digits -- with no grid/task-specific content
# (see its own module docstring); safe under G1. NOA=2 below matches ESAIC
# Theorem 1's fixed centralized joint-of-2 (jax_saic.centralized.NOA), not a
# rendezvous assumption -- every ESAIC application's centralized phase is
# joint-of-2, regardless of the eventual decentralized target N.
NOA = 2
_AGENT_IDX = 1  # matches jax_saic.clustering.value_of_observation's own
# convention: marginalize over agent 0, report for agent 1 ("the last
# agent's own position") -- reused here, not a new assumption.


class CertificationConfig(NamedTuple):
    """All fields default to values that keep certification fully inert
    (cert_enabled=False) -- existing ESAIC behavior is unaffected unless
    this is explicitly turned on. Non-placeholder defaults are the exact
    values given in the certification instruction's spec S4; refine_steps/
    sampling_budget have no universal default (the instruction leaves them
    as "<one existing centralized epoch>" / "<cap on total extra env
    steps>") -- None here means run_esaic_certified() resolves refine_steps
    to the driving CentralizedConfig.ns, and sampling_budget to unbounded
    (gated only by refine_max_rounds) unless explicitly set.
    """

    cert_enabled: bool = False
    cert_mode: str = "bracket"  # "bracket" | "concentration"
    sigma_source: str = "ensemble"  # "ensemble" | "count"
    ensemble_size: int = 5
    z_quantile: float = 1.96
    mc_rollouts: int = 64
    mc_horizon: int = 200  # rollout truncation T'; generic cap, not task-specific
    ucb_scale: float = 1.0
    use_double_q: bool = False
    margin_inflation: float = 1.0
    refine_max_rounds: int = 3
    refine_oversample_frac: float = 0.5
    refine_focus_low_visit_frac: float = 0.5
    # Fraction of F (the uncertified set), by COUNT, kept as focus_states --
    # restricted to F's lowest-N_o member first. Found directly to matter:
    # when most of Omega is uncertified (as it stays for the whole run in a
    # tight-margin/large-U case), F is essentially IDENTICAL to the natural
    # reset distribution (every non-terminal state), so "oversample toward
    # F" was statistically indistinguishable from ordinary training --
    # confirmed by checking F against env.reset's own legal-state set in a
    # real run: they matched exactly, every round. That made the whole
    # refinement mechanism a no-op specifically for the worst-covered
    # (least-visited, worst-value-error) states it was meant to help.
    # Narrowing focus_states to F's least-visited fraction restores genuine
    # concentration regardless of how large F is. 1.0 reproduces the old
    # "focus on all of F" behavior exactly (only a real restriction when
    # < 1.0); 0.5 (default) targets the half of F needing it most.
    refine_steps: Optional[int] = None
    sampling_budget: Optional[int] = None
    uncertified_policy: str = "flag"  # "flag" | "conservative" | "weighted"

    # --- APPEND NEW FIELDS ONLY BELOW THIS LINE, NEVER INSERT ABOVE IT ---
    # This is a NamedTuple: pickle restores instances POSITIONALLY against
    # whatever the CURRENT class definition is, not the one that existed
    # when the pickle was written. Verified directly: inserting a field in
    # the MIDDLE silently shifts every later field's value into the wrong
    # NAME on unpickle (e.g. an old `cert_mode="concentration"` value
    # silently reappearing under a newly-inserted field's name, with
    # `cert_mode` itself quietly reverting to ITS class default) --
    # appending at the end does not have this problem (old pickles just get
    # the new field's default; every existing field's value is unaffected).
    # v_hat_weighting was originally inserted right after cert_enabled and
    # got hit by exactly this bug (silently misread as cert_mode's value in
    # every pickle saved between that insertion and this fix, including
    # centralized_workspace_4x4_1bit{_seed0,_seed1,_seed2}_1000k.pkl and the
    # untagged one) -- moved here, at the true end, specifically so those
    # already-saved pickles unpickle correctly again under the current
    # class (confirmed: append-only never corrupts prior instances).
    v_hat_weighting: str = "visitation"  # "visitation" | "uniform"
    # How V_hat(o) marginalizes over the partner's position. "visitation"
    # (default, backward compatible) = the empirical, per-self-state
    # conditional P(partner|self=o) actually observed during training
    # (_marginalize_joint_q's original design, PORT_NOTES.md S0 item 6).
    # "uniform" = a single fixed distribution, uniform over non-terminal
    # states, matching env.reset's own natural distribution regardless of
    # o -- requested directly after the empirical weighting was found to
    # break geometric symmetries V_hat "should" respect (e.g. two
    # diagonal-reflection-symmetric grid cells getting different V_hat,
    # traced to this weighting choice specifically, not to residual
    # Q-learning error -- see CERTIFICATION.md and uniform_partner_weight()).


CERT_MODES = ("bracket", "concentration")
SIGMA_SOURCES = ("ensemble", "count")
UNCERTIFIED_POLICIES = ("flag", "conservative", "weighted")
V_HAT_WEIGHTINGS = ("visitation", "uniform")


def margins(V_hat: jax.Array, mu: jax.Array) -> jax.Array:
    """Per-observation margin m(o) = 0.5 * (d_runner_up - d_nearest): how
    far V_hat(o) sits from the midpoint of its two nearest medoids, i.e.
    the decision boundary it's closest to. Shape [|Omega|].

    Rationale (spec S1.1): o keeps its correct codeword as long as V*(o)
    stays on the same side of the nearest boundary as V_hat(o); a sufficient
    condition is |V_hat(o) - V*(o)| < m(o). Value error only ever matters
    for the assignment through this ordering (Singh & Yee 1994; Williams &
    Baird 1993) -- so the gate compares to m(o), not a global sup-norm.
    """
    d = jnp.abs(V_hat[:, None] - mu[None, :])  # [|Omega|, B]
    d_sorted = jnp.sort(d, axis=1)
    return 0.5 * (d_sorted[:, 1] - d_sorted[:, 0])


def uniform_partner_weight(n_obs: int, terminal_mask: Optional[jax.Array] = None) -> jax.Array:
    """Uniform distribution over non-terminal observations -- the actual
    natural episode-reset distribution (env.reset draws i.i.d. uniformly
    over legal/non-terminal states), as an alternative to
    _marginalize_joint_q's DEFAULT empirical, visitation-based
    P(partner|self). Same weight for every self-state o (not one
    distribution per o like the empirical version), so it also restores
    the environment's own geometric symmetries in V_hat (e.g. two
    diagonal-reflection-symmetric observations get IDENTICAL V_hat under
    this weighting, given the same Q-table -- verified directly: exact
    value iteration's V_o under this exact weighting scheme is equal to
    6 decimal places for such a pair; it is NOT generally equal under the
    empirical/visitation weighting, since that depends on one particular
    training run's own asymmetric visitation history, not the problem's
    geometry). Terminal observations get weight 0 (env.reset never starts
    an agent there); n_obs, not grid-specific, per G1.
    """
    if terminal_mask is None:
        return jnp.full((n_obs,), 1.0 / n_obs)
    non_terminal = ~terminal_mask
    return jnp.where(non_terminal, 1.0 / jnp.sum(non_terminal), 0.0)


def _marginalize_joint_q(
    q_joint: jax.Array, N_joint: jax.Array, n_obs: int, n_actions: int,
    partner_weight: Optional[jax.Array] = None,
) -> jax.Array:
    """Marginalize a joint (NOA=2) Q-table [n_obs^2, n_actions^2] down to
    agent _AGENT_IDX's own (o, a) slice, shape [n_obs, n_actions].

    NOTE, found by testing (test_bracket_ordering caught it): this does
    NOT use jax_saic.clustering.value_of_observation()'s own weighting
    verbatim. That weighting (each joint state's "probability" approximated
    by its max-visited-action count, divided by the TOTAL visitation count
    across the WHOLE joint table) does not sum to 1 over the partner's
    position for a fixed observation o -- it's explicitly flagged in
    PORT_NOTES.md SS4.2/SS11.5 as "sum_q_MultiAgent.m's approximation,
    verbatim", not a real conditional probability. Fixed here to a PROPERLY
    normalized empirical conditional distribution instead (the DEFAULT,
    partner_weight=None): P(partner state = j | self state = i) =
    N_state(j,i) / sum_j' N_state(j',i), a genuine probability distribution
    over the partner's position for each o. Still fully generic (n_obs/
    n_actions only, no task-specific content) -- see CERTIFICATION.md for
    why this deviates from S0's original design note to reuse
    value_of_observation's exact weighting.

    partner_weight: optional [n_obs] distribution over the partner's
    position, used INSTEAD of the empirical per-self-state conditional
    above when given (SAME distribution for every self-state o, unlike the
    empirical default) -- e.g. uniform_partner_weight() above, requested
    directly after the empirical weighting was found to break geometric
    symmetries V_hat "should" have (diagonal-reflection-symmetric grid
    cells getting different V_hat, traced to this weighting choice, not to
    Q-learning imprecision -- see CERTIFICATION.md).

    Array sizes here (n_obs^2, n_actions^2) match jax_saic.centralized.
    train()'s own qp_table/N_table shape exactly -- fixed by NOA=2, not
    scaling with the decentralized target N (see module docstring, G2).
    """
    agent0_state, agent1_state = jnp.meshgrid(jnp.arange(n_obs), jnp.arange(n_obs), indexing="ij")
    main_state_grid = agent0_state + n_obs * agent1_state  # [n_obs, n_obs], agent-0-fastest

    q_full = q_joint[main_state_grid]  # [n_obs(a0), n_obs(a1), n_actions^2]

    agent0_act, agent1_act = jnp.meshgrid(jnp.arange(n_actions), jnp.arange(n_actions), indexing="ij")
    main_action_grid = agent0_act + n_actions * agent1_act  # [n_actions, n_actions]

    q_reshaped = q_full[:, :, main_action_grid]  # [n_obs(a0), n_obs(a1), n_actions(a0), n_actions(a1)]

    if partner_weight is None:
        N_full = N_joint[main_state_grid]
        N_state = jnp.sum(N_full, axis=-1)  # [n_obs(a0), n_obs(a1)] -- total visits to this joint STATE
        prob_table = N_state / jnp.sum(N_state, axis=0, keepdims=True)  # normalized PER agent1-state (sums to 1 over axis 0)
    else:
        prob_table = jnp.broadcast_to(partner_weight[:, None], (n_obs, n_obs))  # same weight for every self-state

    partner_max_q = jnp.max(q_reshaped, axis=2)  # max over agent0's action -> [n_obs(a0), n_obs(a1), n_actions(a1)]
    weighted = partner_max_q * prob_table[:, :, None]
    return jnp.sum(weighted, axis=0)  # marginalize over agent0's state -> [n_obs, n_actions]


def bracket_q_ub(
    qp_table_joint: jax.Array,
    N_table_joint: jax.Array,
    n_obs: int,
    n_actions: int,
    ucb_scale: float,
    rmax: float,
    gamma: float,
) -> jax.Array:
    """Q_ub(o, a): optimistic upper Q over Ω x A (spec S1.2(a)). Inflates
    the joint Q-table with a Hoeffding-style UCB bonus (discounted variant;
    Jin, Allen-Zhu, Bubeck & Jordan 2018) so Q_joint_ub >= Q*_joint w.h.p.,
    then marginalizes down to a single agent's (o, a) via
    _marginalize_joint_q -- the width brackets V* from above by
    construction (spec S1.2(a)). Shape [n_obs, n_actions].
    """
    bonus = (
        ucb_scale
        * rmax
        / (1.0 - gamma)
        * jnp.sqrt(jnp.log(jnp.sum(N_table_joint) + 1.0) / N_table_joint)
    )
    q_joint_ub = qp_table_joint + bonus
    return _marginalize_joint_q(q_joint_ub, N_table_joint, n_obs, n_actions)


def _conditional_partner_distribution(N_joint: jax.Array, n_obs: int) -> jax.Array:
    """P(partner state = j | self state = i) -- the SAME empirical
    conditional weighting _marginalize_joint_q uses for V_hat (visitation-
    based, per PORT_NOTES.md S0 item 6's deliberate "estimated empirically
    from centralized-training rollout visitation frequency, not assumed
    uniform" choice). Shape [n_obs(partner), n_obs(self)], each column sums
    to 1. Factored out so bracket_v_pi_lower's MC rollouts start from the
    SAME distribution V_hat's marginalization is defined against -- using
    env.reset_to's environment-NATURAL (uniform) partner distribution
    instead would compare V_hat and V_pi_lower under two different
    definitions of "the other agent's position given mine", which testing
    (test_bracket_ordering) showed breaks the bracket's own soundness
    claim (V_pi_lower(o) should not systematically exceed V_hat(o) if both
    describe the same quantity).
    """
    agent0_state, agent1_state = jnp.meshgrid(jnp.arange(n_obs), jnp.arange(n_obs), indexing="ij")
    main_state_grid = agent0_state + n_obs * agent1_state
    N_full = N_joint[main_state_grid]
    N_state = jnp.sum(N_full, axis=-1)  # [n_obs(a0), n_obs(a1)]
    return N_state / jnp.sum(N_state, axis=0, keepdims=True)


def _mc_rollout_return(
    rng, env: EnvSpec, qp_table_joint: jax.Array, prob_partner_given_self: jax.Array, o, horizon: int, gamma: float
):
    """One MC rollout of the CURRENT GREEDY joint policy (argmax of
    qp_table_joint), started with agent _AGENT_IDX at `o` and the partner
    sampled from prob_partner_given_self[:, o] (see
    _conditional_partner_distribution) -- NOT env.reset_to, which would use
    the environment's natural (typically uniform) distribution instead;
    see that function's docstring for why. Every EnvSpec-conforming
    environment's joint_obs is by construction the complete state env.step
    needs, so building it directly (rather than via a reset primitive) is
    still fully generic. Returns the discounted sum of REPORTED (env.step's
    own, not shaped) rewards, truncated at `horizon` steps. jax.lax.scan,
    no Python loop.
    """
    rng_partner, rng_roll = jax.random.split(rng)
    partner = jax.random.categorical(rng_partner, jnp.log(prob_partner_given_self[:, o] + 1e-12))
    joint_obs0 = jnp.zeros((env.n_agents,), dtype=jnp.int32).at[0].set(partner).at[_AGENT_IDX].set(o)

    def step_fn(carry, step_rng):
        joint_obs, discount, total, done = carry
        main_idx = flatten_mixed_radix(joint_obs, base=env.n_obs)
        joint_action_flat = jnp.argmax(qp_table_joint[main_idx])
        joint_action = unflatten_mixed_radix(joint_action_flat, base=env.n_actions, k=env.n_agents)
        next_obs, reward, terminal = env.step(joint_obs, joint_action, step_rng)
        new_total = total + jnp.where(done, 0.0, discount * reward)
        new_discount = discount * gamma
        new_done = done | (terminal > 0)
        next_obs = jnp.where(done, joint_obs, next_obs)
        return (next_obs, new_discount, new_total, new_done), None

    step_rngs = jax.random.split(rng_roll, horizon)
    init = (joint_obs0, jnp.array(1.0), jnp.array(0.0), jnp.array(False))
    (_, _, total, _), _ = jax.lax.scan(step_fn, init, step_rngs)
    return total


def bracket_v_pi_lower(
    rng: jax.Array,
    env: EnvSpec,
    qp_table_joint: jax.Array,
    N_table_joint: jax.Array,
    mc_rollouts: int,
    mc_horizon: int,
    gamma: float,
) -> jax.Array:
    """V_pi_lower(o): unbiased MC estimate of the CURRENT GREEDY policy's
    value at every o in Ω, averaged over `mc_rollouts` trajectories each
    (spec S1.2(a)), partner sampled from the same empirical conditional
    distribution V_hat's own marginalization uses (see
    _conditional_partner_distribution). V_pi_lower(o) <= V*(o) always (a
    greedy policy w.r.t. a possibly-imperfect Q is suboptimal), so this
    brackets V* from below. vmapped over both observations and rollouts --
    no Python-level loop over Ω. Shape [n_obs].
    """
    prob_partner_given_self = _conditional_partner_distribution(N_table_joint, env.n_obs)

    def rollouts_for_one_obs(o, obs_rng):
        rollout_rngs = jax.random.split(obs_rng, mc_rollouts)
        totals = jax.vmap(
            lambda r: _mc_rollout_return(r, env, qp_table_joint, prob_partner_given_self, o, mc_horizon, gamma)
        )(rollout_rngs)
        return jnp.mean(totals)

    obs = jnp.arange(env.n_obs)
    obs_rngs = jax.random.split(rng, env.n_obs)
    return jax.vmap(rollouts_for_one_obs)(obs, obs_rngs)


def bracket_certificate(
    rng: jax.Array,
    env: EnvSpec,
    qp_table_joint: jax.Array,
    N_table_joint: jax.Array,
    config: CertificationConfig,
    rmax: float,
    gamma: float,
) -> jax.Array:
    """U_bracket(o) = max_a Q_ub(o,a) - V_pi_lower(o), clamped >= 0 (spec
    S1.2(a)) -- avoids the Bellman residual entirely (its single-sample
    square is biased by the double-sampling term, Baird 1995). Shape
    [n_obs].
    """
    Q_ub = bracket_q_ub(
        qp_table_joint, N_table_joint, env.n_obs, env.n_actions, config.ucb_scale, rmax, gamma
    )
    V_pi_lower = bracket_v_pi_lower(
        rng, env, qp_table_joint, N_table_joint, config.mc_rollouts, config.mc_horizon, gamma
    )
    return jnp.clip(jnp.max(Q_ub, axis=-1) - V_pi_lower, 0.0)


def marginalize_joint_n(N_joint: jax.Array, n_obs: int) -> jax.Array:
    """N_o(o): total visitation count for agent _AGENT_IDX at observation o,
    marginalized over agent 0's position and BOTH agents' actions (plain
    sum, not max) -- same convention as
    jax_saic.clustering.value_of_observation's N_o, generalized from
    n2->n_obs. Shape [n_obs].
    """
    agent0_state, agent1_state = jnp.meshgrid(jnp.arange(n_obs), jnp.arange(n_obs), indexing="ij")
    main_state_grid = agent0_state + n_obs * agent1_state
    N_full = N_joint[main_state_grid]  # [n_obs(a0), n_obs(a1), n_actions_joint]
    return jnp.sum(N_full, axis=(0, -1))


def marginalize_joint_v(
    q_joint: jax.Array, N_joint: jax.Array, n_obs: int, n_actions: int,
    partner_weight: Optional[jax.Array] = None,
) -> jax.Array:
    """V_hat(o) = max_a of _marginalize_joint_q -- the certification
    module's own generic replacement for
    jax_saic.clustering.value_of_observation(), which isn't reusable here:
    it takes a grid side-length `n` and internally assumes n_obs == n*n,
    a hidden grid-specific assumption (G1). This version is parameterized
    directly by n_obs/n_actions. Shape [n_obs].

    partner_weight: forwarded to _marginalize_joint_q -- None (default)
    keeps the empirical, visitation-based per-self-state conditional;
    pass uniform_partner_weight(n_obs, env.terminal_mask) for a fixed,
    self-state-independent weighting instead (see that function and
    CertificationConfig.v_hat_weighting).
    """
    return jnp.max(_marginalize_joint_q(q_joint, N_joint, n_obs, n_actions, partner_weight=partner_weight), axis=-1)


def concentration_sigma_ensemble(rng, env: EnvSpec, gc_config, ensemble_size: int):
    """sigma_hat(o) = std over `ensemble_size` independently-seeded
    generic_centralized.train() reruns' V_hat(o) (spec S1.2(b)) -- captures
    approximation/training-variance uncertainty, not just within-run
    sampling noise. Returns (sigma_hat, ensemble_mean_V_hat), both [n_obs].

    Cost: O(ensemble_size) full centralized retrains -- independent of N
    (each retrain is itself the same O(1)-in-N joint-of-2 cost as
    jax_saic.centralized.train()). The Python loop here is over ensemble
    members, not observations -- G2's "no Python loop over Omega" is about
    per-observation work, which stays vectorized inside each retrain and
    inside marginalize_joint_v.
    """
    from esaic import generic_centralized  # local import: avoids a module-load-order

    # dependency between certification.py and generic_centralized.py (the
    # latter already imports EnvSpec from esaic.env_spec, not from here).
    keys = jax.random.split(rng, ensemble_size)
    V_hats = []
    for k in keys:
        qp, N_emerged, _steps = generic_centralized.train(gc_config, env, k)
        V_hats.append(marginalize_joint_v(qp, N_emerged, env.n_obs, env.n_actions))
    V_stack = jnp.stack(V_hats, axis=0)  # [ensemble_size, n_obs]
    return jnp.std(V_stack, axis=0), jnp.mean(V_stack, axis=0)


def concentration_sigma_count(N_o: jax.Array, rmax: float, gamma: float, c: float = 1.0) -> jax.Array:
    """sigma_hat(o) = c * Rmax / ((1-gamma) * sqrt(n(o))) (spec S1.2(b)):
    the count-based alternative, grounded in tabular ell-infinity theory
    (Wainwright 2019; Li, Wei, Chi, Gu & Chen 2022; Qu & Wierman 2020) --
    entrywise error scales with per-entry visitation, governed by mu_min
    (the minimum occupancy). N_o: [n_obs] visitation counts. No training
    run needed -- cheap, reuses whatever N_o the caller already has.
    """
    return c * rmax / ((1.0 - gamma) * jnp.sqrt(jnp.maximum(N_o, 1.0)))


def concentration_b_max(rng, env: EnvSpec, gc_config) -> jax.Array:
    """Maximization-bias budget (spec S1.2(b)), when use_double_q=False:
    estimated as max(0, V_singleQ(o) - V_doubleQ(o)) from a short PAIRED
    run (same seed, same episode count, only use_double_q differs) --
    single-Q's known overestimation bias should show up as
    V_singleQ >= V_doubleQ on average. gc_config.use_double_q is ignored
    (both variants are run explicitly); pass a config with a modest `ns`
    since this is meant to be a short diagnostic run, not full training.
    """
    from esaic import generic_centralized

    qp_single, N_single, _s1 = generic_centralized.train(gc_config._replace(use_double_q=False), env, rng)
    qp_double, N_double, _s2 = generic_centralized.train(gc_config._replace(use_double_q=True), env, rng)
    V_single = marginalize_joint_v(qp_single, N_single, env.n_obs, env.n_actions)
    V_double = marginalize_joint_v(qp_double, N_double, env.n_obs, env.n_actions)
    return jnp.clip(V_single - V_double, 0.0)


def concentration_certificate(
    rng: jax.Array,
    env: EnvSpec,
    gc_config,
    config: CertificationConfig,
    rmax: float,
    gamma: float,
    N_o: Optional[jax.Array] = None,
) -> jax.Array:
    """U_conc(o) = z_{1-alpha} * sigma_hat(o) + b_max(o) (spec S1.2(b)).
    gc_config: a generic_centralized.GenericCentralizedConfig used for any
    training this needs (ensemble members, or the paired single/double-Q
    b_max run) -- NOT the same object as `config` (CertificationConfig).
    """
    if config.use_double_q:
        b_max = jnp.zeros(env.n_obs)
    else:
        warnings.warn(
            "cert_mode='concentration' with use_double_q=False: b_max is "
            "estimated from a short paired single/double-Q run (spec "
            "S1.2(b)) rather than known to be zero; consider "
            "use_double_q=True to eliminate this budget instead of "
            "estimating it."
        )
        rng, k_bmax = jax.random.split(rng)
        b_max = concentration_b_max(k_bmax, env, gc_config)

    if config.sigma_source == "ensemble":
        rng, k_ens = jax.random.split(rng)
        sigma_hat, _ensemble_mean_V_hat = concentration_sigma_ensemble(k_ens, env, gc_config, config.ensemble_size)
    elif config.sigma_source == "count":
        if N_o is None:
            raise ValueError("sigma_source='count' requires N_o (visitation counts)")
        sigma_hat = concentration_sigma_count(N_o, rmax, gamma)
    else:
        raise ValueError(f"Unknown sigma_source {config.sigma_source!r}; expected one of {SIGMA_SOURCES}")

    return config.z_quantile * sigma_hat + b_max


def certify(
    V_hat: jax.Array,
    mu: jax.Array,
    U: jax.Array,
    delta_medoid: jax.Array,
    config: CertificationConfig,
):
    """The gate (spec S1.3/S1.4). certified[o] == True means o's codeword
    provably equals its V*-codeword (w.h.p.) -- the uncertified set
    F = {o : ~certified[o]} is what refinement targets.

    delta_medoid: scalar, max medoid shift between the last two clustering
    rounds (0.0 on the first round / when refinement isn't running) --
    m(o) is computed from V_hat-based medoids, not V*-based ones, so if
    points move under refinement, medoids move too; this inflates the
    margin defensively against that second-order effect.
    """
    m = margins(V_hat, mu)
    m_eff = jnp.clip(m - config.margin_inflation * delta_medoid, 0.0)
    certified = U < m_eff
    return certified, m_eff


def _compute_certificate(rng, env, gc_config, config, qp_table, N_table_emerged, N_o, rmax, gamma):
    if config.cert_mode == "bracket":
        return bracket_certificate(rng, env, qp_table, N_table_emerged, config, rmax, gamma)
    elif config.cert_mode == "concentration":
        return concentration_certificate(rng, env, gc_config, config, rmax, gamma, N_o=N_o)
    else:
        raise ValueError(f"Unknown cert_mode {config.cert_mode!r}; expected one of {CERT_MODES}")


def _ag_states_from_cluster_id(cluster_id: np.ndarray, k: int) -> np.ndarray:
    """Rebuild the padded ag_states format ([k, max_size] int32, -1 pad)
    from a plain per-observation cluster_id array -- same bincount+fill
    pattern jax_saic.clustering.cluster_states() itself uses internally,
    needed here because uncertified_policy='conservative' reassigns some
    observations after the normal nearest-medoid clustering call.
    """
    counts = np.bincount(cluster_id, minlength=k)
    max_size = int(counts.max()) if len(counts) else 0
    ag_states = np.full((k, max_size), -1, dtype=np.int32)
    fill = np.zeros(k, dtype=np.int64)
    for state, cid in enumerate(cluster_id):
        ag_states[cid, fill[cid]] = state
        fill[cid] += 1
    return ag_states


def _apply_uncertified_policy(V_hat, mu, cluster_id, ag_states, certified, U, config, inf_bits, cluster_method):
    """Spec S3 'Residual uncertainty handling': what to do with
    observations still uncertified after refinement stops. Returns
    (cluster_id, ag_states), possibly unchanged (mode='flag').
    """
    k = 2**inf_bits
    if config.uncertified_policy == "flag":
        return cluster_id, ag_states

    if config.uncertified_policy == "conservative":
        # Reassign each UNCERTIFIED o to the LOWER-value of its two nearest
        # medoids -- fail toward under-, not over-estimation. Certified o's
        # keep their normal nearest-medoid assignment.
        d = jnp.abs(V_hat[:, None] - mu[None, :])
        order = jnp.argsort(d, axis=1)
        nearest_idx = order[:, 0]
        runner_up_idx = order[:, 1]
        lower_of_two = jnp.where(mu[nearest_idx] <= mu[runner_up_idx], nearest_idx, runner_up_idx)
        new_cluster_id = jnp.where(certified, cluster_id, lower_of_two)
        new_ag_states = _ag_states_from_cluster_id(np.asarray(new_cluster_id), k)
        return new_cluster_id, new_ag_states

    if config.uncertified_policy == "weighted":
        # Re-cluster with 1/U(o) as the sample weight (replacing N_o) so
        # unresolved (high-U) points pull less on the weighted median.
        weight = 1.0 / jnp.maximum(U, 1e-8)
        new_ag_states, new_mu = cluster_states(
            np.asarray(V_hat), np.asarray(weight), inf_bits, method=cluster_method, return_medoids=True
        )
        new_mu_j = jnp.asarray(new_mu)
        new_cluster_id = jnp.argmin(jnp.abs(V_hat[:, None] - new_mu_j[None, :]), axis=1)
        return new_cluster_id, new_ag_states

    raise ValueError(f"Unknown uncertified_policy {config.uncertified_policy!r}; expected one of {UNCERTIFIED_POLICIES}")


def _cluster(V_hat: jax.Array, N_o: jax.Array, inf_bits: int, cluster_method: str, terminal_mask: Optional[jax.Array]):
    """cluster_states(), optionally excluding terminal_mask==True
    observations from the k-median OPTIMIZATION itself -- their value
    estimate, however imprecise, shouldn't distort where boundaries fall
    for observations that actually need a codeword (see
    EnvSpec.terminal_mask's docstring: no control decision is ever made
    conditioned on a terminal observation). Terminal observations still
    get a valid slot in the returned ag_states/cluster_id, assigned
    post-hoc to their nearest medoid by value (not part of the k-median
    objective) -- downstream code (jax_saic.train.s_aggregate) needs
    every observation covered by SOME cluster.
    """
    V_hat_np = np.asarray(V_hat)
    N_o_np = np.asarray(N_o)

    if terminal_mask is None:
        ag_states, mu_np = cluster_states(V_hat_np, N_o_np, inf_bits, method=cluster_method, return_medoids=True)
    else:
        non_terminal_idx = np.where(~np.asarray(terminal_mask))[0]
        _discard, mu_np = cluster_states(
            V_hat_np[non_terminal_idx], N_o_np[non_terminal_idx], inf_bits,
            method=cluster_method, return_medoids=True,
        )

    mu = jnp.asarray(mu_np)
    cluster_id = jnp.argmin(jnp.abs(V_hat[:, None] - mu[None, :]), axis=1)  # nearest medoid for EVERY
    # observation, including terminal ones when excluded above -- this is
    # what makes their assignment "post-hoc": not part of the k-median cost.
    if terminal_mask is not None:
        ag_states = _ag_states_from_cluster_id(np.asarray(cluster_id), 2**inf_bits)
    return ag_states, mu, cluster_id


def _certify_round(rng, env, gc_config, config, qp_table, N_table_emerged, V_hat, mu, N_o, delta_medoid, rmax, gamma):
    """One certificate + gate pass, with terminal observations (if
    env.terminal_mask is set) forced certified=True -- there's nothing to
    certify for them (no decision is ever made conditioned on them), and
    leaving them uncertified would waste refinement budget chasing a
    precision improvement that can never matter.
    """
    U = _compute_certificate(rng, env, gc_config, config, qp_table, N_table_emerged, N_o, rmax, gamma)
    certified, m_eff = certify(V_hat, mu, U, delta_medoid, config)
    if env.terminal_mask is not None:
        certified = certified | env.terminal_mask
    F = jnp.where(~certified)[0]
    return U, certified, m_eff, F


class CertificationResult(NamedTuple):
    """Spec S3 step 5's emitted result. ag_states is
    jax_saic.clustering.cluster_states()'s own padded format, ready to feed
    directly into jax_saic.train.s_aggregate()/run_phase1() for the
    rendezvous task, or the equivalent decentralized-training entry point
    for any other EnvSpec-conforming task.
    """

    V_hat: jax.Array
    N_o: jax.Array
    mu: jax.Array
    cluster_id: jax.Array
    ag_states: np.ndarray
    certified: jax.Array
    m_eff: jax.Array
    U: jax.Array
    F: jax.Array
    round_stats: tuple


def run_esaic_certified(
    rng: jax.Array,
    env: EnvSpec,
    gc_config,
    inf_bits: int,
    config: CertificationConfig,
    rmax: float,
    gamma: float,
    cluster_method: str = "kmedian",
    train_fn=None,
) -> CertificationResult:
    """Orchestration (spec S3): initial centralized train -> cluster ->
    certify -> loop, oversampling F's lowest-visitation fraction (see
    CertificationConfig.refine_focus_low_visit_frac) via focus-biased
    retraining (train_fn's focus_states/focus_frac hook), re-cluster,
    re-certify, until F is empty, config.sampling_budget is exhausted (in
    real extra env-step units, tracked via train_fn's returned step count),
    or config.refine_max_rounds is reached. Diagnostics logged per round
    (spec S5) in round_stats. Residual uncertainty after stopping is
    handled per config.uncertified_policy (spec S3).

    train_fn defaults to esaic.generic_centralized.train -- the only
    trainer that's actually generic over EnvSpec (see that module's
    docstring for why jax_saic.centralized.train can't be reused here).
    A different train_fn with the same signature
    (gc_config, env, rng, focus_states=None, focus_frac=0.0,
    init_qp_table=None, init_N_table_emerged=None) ->
    (qp_table, N_table_emerged, steps) may be substituted by a caller that
    specifically wants a different trainer. Refinement rounds pass the
    previous round's (qp_table, N_table_emerged) forward as
    init_qp_table/init_N_table_emerged so each round continues training
    instead of restarting from scratch -- found directly to matter: without
    this, each round's "refined" table was a from-scratch run of only
    refine_steps episodes, discarding the (typically much larger) prior
    training rather than building on it, which made certification
    unreachable through refinement regardless of how many rounds ran.
    """
    if train_fn is None:
        from esaic import generic_centralized

        train_fn = generic_centralized.train

    if config.v_hat_weighting not in V_HAT_WEIGHTINGS:
        raise ValueError(f"Unknown v_hat_weighting {config.v_hat_weighting!r}; expected one of {V_HAT_WEIGHTINGS}")
    partner_weight = (
        uniform_partner_weight(env.n_obs, env.terminal_mask) if config.v_hat_weighting == "uniform" else None
    )

    rng, k_train = jax.random.split(rng)
    qp_table, N_table_emerged, _initial_steps = train_fn(gc_config, env, k_train)
    # Only refinement-round steps count against sampling_budget (spec S3's
    # "extra centralized-training sampling") -- the initial run is the base
    # cost every certification pass pays regardless of refinement.
    total_extra_steps = 0

    V_hat = marginalize_joint_v(qp_table, N_table_emerged, env.n_obs, env.n_actions, partner_weight=partner_weight)
    N_o = marginalize_joint_n(N_table_emerged, env.n_obs)

    ag_states, mu, cluster_id = _cluster(V_hat, N_o, inf_bits, cluster_method, env.terminal_mask)

    delta_medoid = jnp.array(0.0)
    round_stats = []
    refine_steps = config.refine_steps if config.refine_steps is not None else gc_config.ns

    certified = jnp.zeros(env.n_obs, dtype=bool)
    m_eff = jnp.zeros(env.n_obs)
    U = jnp.zeros(env.n_obs)

    for round_idx in range(config.refine_max_rounds):
        rng, k_cert = jax.random.split(rng)
        U, certified, m_eff, F = _certify_round(
            k_cert, env, gc_config, config, qp_table, N_table_emerged, V_hat, mu, N_o, delta_medoid, rmax, gamma
        )

        round_stats.append(
            {
                "round": round_idx,
                "frac_certified": float(jnp.mean(certified)),
                "n_uncertified": int(F.shape[0]),
                "certified": np.asarray(certified),  # per-state boolean, this round -- added
                # so callers can see WHICH states (de)certify round to round, not just the
                # aggregate frac_certified/n_uncertified (diagnostic only, no algorithmic effect).
                "F": np.asarray(F),  # per-round uncertified state indices, same reason
                "min_m_eff": float(jnp.min(m_eff)),
                "mean_U": float(jnp.mean(U)),
                "median_U": float(jnp.median(U)),
                "delta_medoid": float(delta_medoid),
                "extra_env_steps_so_far": total_extra_steps,
            }
        )

        budget_exhausted = config.sampling_budget is not None and total_extra_steps >= config.sampling_budget
        if F.shape[0] == 0 or budget_exhausted:
            break

        # Restrict focus_states to F's lowest-N_o fraction (see
        # CertificationConfig.refine_focus_low_visit_frac's docstring for
        # why: F alone degenerates to a no-op oversample when it's most of
        # Omega). Plain numpy -- run_esaic_certified's loop is eager Python,
        # not jitted, so F.shape[0] is already a concrete int here.
        F_np, N_o_np = np.asarray(F), np.asarray(N_o)
        n_focus = max(1, int(np.ceil(config.refine_focus_low_visit_frac * F_np.shape[0])))
        focus_order = np.argsort(N_o_np[F_np])  # indices INTO F_np, ascending by visitation
        focus_states = jnp.asarray(F_np[focus_order][:n_focus])

        rng, k_refine = jax.random.split(rng)
        qp_table, N_table_emerged, steps_this_round = train_fn(
            gc_config._replace(ns=refine_steps),
            env,
            k_refine,
            focus_states=focus_states.astype(jnp.int32),
            focus_frac=config.refine_oversample_frac,
            init_qp_table=qp_table,
            init_N_table_emerged=N_table_emerged,
        )
        total_extra_steps += steps_this_round

        V_hat = marginalize_joint_v(qp_table, N_table_emerged, env.n_obs, env.n_actions, partner_weight=partner_weight)
        N_o = marginalize_joint_n(N_table_emerged, env.n_obs)

        new_ag_states, new_mu, new_cluster_id = _cluster(V_hat, N_o, inf_bits, cluster_method, env.terminal_mask)
        # medoid values aren't index-aligned across rounds (sorted-by-value,
        # not identity-tracked) -- comparing sorted arrays is the correct
        # "how far did the SET of medoids move" measure.
        delta_medoid = jnp.max(jnp.abs(jnp.sort(new_mu) - jnp.sort(mu)))
        mu = new_mu
        ag_states = new_ag_states
        cluster_id = new_cluster_id

    # If the loop exited via refine_max_rounds (not an empty-F/budget break),
    # `certified`/`U`/`m_eff` above are one retrain stale -- always finish
    # with one consistent final gate pass against the latest V_hat/mu.
    rng, k_final = jax.random.split(rng)
    U, certified, m_eff, F = _certify_round(
        k_final, env, gc_config, config, qp_table, N_table_emerged, V_hat, mu, N_o, delta_medoid, rmax, gamma
    )

    cluster_id, ag_states = _apply_uncertified_policy(
        V_hat, mu, cluster_id, ag_states, certified, U, config, inf_bits, cluster_method
    )

    return CertificationResult(
        V_hat=V_hat,
        N_o=N_o,
        mu=mu,
        cluster_id=cluster_id,
        ag_states=ag_states,
        certified=certified,
        m_eff=m_eff,
        U=U,
        F=F,
        round_stats=tuple(round_stats),
    )

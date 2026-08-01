"""Value-of-observation + state-aggregation clustering. Ports (PORT_NOTES.md
SS4.2/SS4.3):

  Fully Centralized - MultiAgent/sum_q_MultiAgent.m   -> value_of_observation()
  Fully Centralized - MultiAgent/aggregate_states_SAIC.m -> cluster_states()

Both operate on the noa=2 centralized-training output (centralized.py).
This is a one-shot preprocessing step over an (n^2,)-sized state space, run
once per pipeline execution -- not part of the RL training hot path -- so
it's implemented with plain NumPy rather than forced into jax.jit.
"""
from __future__ import annotations

import numpy as np

NOA = 2  # matches centralized.py -- this module only ever consumes its output


def value_of_observation(qp_table, N_table_emerged, n: int):
    """Port of sum_q_MultiAgent.m, specialized to noa=2 (PORT_NOTES.md SS0.8:
    the centralized phase never runs at any other noa in this pipeline).

    qp_table, N_table_emerged: (n^2 * n^2, 25) arrays, main_ps0-indexed
    (agent-0-fastest flattening, see indexing.py) x flattened joint action.

    Returns (V_o, N_o), each shape (n^2,): the value/visitation-count of
    "the last agent's" (agent index 1, 0-indexed) own position, marginalized
    over agent 0's position -- reused for every agent by the rendezvous
    problem's symmetry (PORT_NOTES.md SS4.2).
    """
    qp_table = np.asarray(qp_table)
    N_table_emerged = np.asarray(N_table_emerged)
    n2 = n * n

    # Unflatten the state axis only (not the action axis -- max() over the
    # still-flattened 25-action axis is all sum_q_MultiAgent.m needs).
    # main_ps0 = ps0_agent0 + n2*ps0_agent1 (agent 0 fastest, indexing.py).
    # Built via an explicit index grid rather than a reshape(order='F') to
    # avoid any ambiguity about reshape memory-order conventions.
    agent0_idx, agent1_idx = np.meshgrid(np.arange(n2), np.arange(n2), indexing="ij")
    main_ps0_grid = agent0_idx + n2 * agent1_idx  # shape (n2, n2)

    N_table = N_table_emerged[main_ps0_grid]  # shape (n2, n2, 25)
    qp_table_full = qp_table[main_ps0_grid]  # shape (n2, n2, 25)

    # sum_q_MultiAgent.m's approximation, verbatim (PORT_NOTES.md SS4.2):
    # "probability" of a joint state is approximated by the visitation count
    # of its MOST-VISITED action (not summed over actions), on the stated
    # assumption that post-convergence only the greedy action is ever taken.
    prob_table = np.max(N_table, axis=-1) / np.sum(N_table_emerged)  # (n2, n2)
    v_o_mat = np.max(qp_table_full, axis=-1)  # (n2, n2) -- V*(joint state), eq. 42
    v_o_weighted = v_o_mat * prob_table

    # Marginalize over agent 0's position (axis 0), keeping agent 1's
    # position fixed -- "computed only for the last agent index" (SS4.2).
    V_o = np.sum(v_o_weighted, axis=0)  # (n2,)
    # N_o is a PLAIN sum (not max) over agent 0's position AND the action
    # axis of the raw visitation-count table -- a different marginalization
    # than prob_table's, used later for the clustering replication weight.
    N_o = np.sum(N_table, axis=(0, -1))  # (n2,)

    return V_o, N_o


def _weighted_median_cost(sorted_vals, sorted_weights, prefix_w, prefix_wv, l, r):
    """Cost of putting sorted_vals[l:r+1] (weights sorted_weights[l:r+1])
    into one cluster: sum w_i*|v_i - median|, median = the weighted median
    of that (already value-sorted) slice."""
    w_lr = prefix_w[r + 1] - prefix_w[l]
    target = prefix_w[l] + w_lr / 2.0
    # smallest m in [l, r] such that cumulative weight up to m >= target
    m = l
    while m < r and prefix_w[m + 1] < target:
        m += 1
    median = sorted_vals[m]
    left_cost = median * (prefix_w[m + 1] - prefix_w[l]) - (prefix_wv[m + 1] - prefix_wv[l])
    right_cost = (prefix_wv[r + 1] - prefix_wv[m + 1]) - median * (prefix_w[r + 1] - prefix_w[m + 1])
    return left_cost + right_cost


def _exact_weighted_kmedian(values: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """Exact solve of eq. 13 (SAIC paper): partition `values` (each weighted
    by `weights`, e.g. visitation counts) into k contiguous-in-sorted-order
    groups minimizing total weighted L1 cost, via DP. For 1-D data under an
    L1 cost, the optimal partition is always contiguous in sorted order and
    each group's optimal center is its weighted median -- both standard
    results, used here rather than a generic (approximate) solver.

    Returns the k medoid VALUES (not just labels) -- see PORT_NOTES.md SS4.3
    on why keeping the medoid values, not just labels, matters.
    """
    order = np.argsort(values)
    sv = values[order]
    sw = weights[order].astype(np.float64)
    m = len(sv)
    prefix_w = np.concatenate([[0.0], np.cumsum(sw)])
    prefix_wv = np.concatenate([[0.0], np.cumsum(sw * sv)])

    cost = np.full((m + 1,), np.inf)
    # cost of putting the whole prefix [0, r) into ONE group -- j=1 base case
    cost[0] = 0.0
    group_of = [[] for _ in range(m + 1)]
    # dp[j][r] = min cost of partitioning sv[0:r] into j groups
    dp = np.full((k + 1, m + 1), np.inf)
    split = np.full((k + 1, m + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    for j in range(1, k + 1):
        for r in range(1, m + 1):
            best = np.inf
            best_l = -1
            for l in range(0, r):
                if dp[j - 1, l] == np.inf:
                    continue
                c = dp[j - 1, l] + _weighted_median_cost(sv, sw, prefix_w, prefix_wv, l, r - 1)
                if c < best:
                    best = c
                    best_l = l
            dp[j, r] = best
            split[j, r] = best_l

    # Backtrack to find each group's weighted median (the medoid value).
    medoids = []
    r = m
    for j in range(k, 0, -1):
        l = split[j, r]
        if l < 0:
            # Fewer than k non-empty groups fit (e.g. k > number of distinct
            # values) -- pad with a copy of the last real medoid found.
            medoids.append(medoids[-1] if medoids else sv[-1])
            continue
        target = prefix_w[l] + (prefix_w[r] - prefix_w[l]) / 2.0
        mm = l
        while mm < r - 1 and prefix_w[mm + 1] < target:
            mm += 1
        medoids.append(sv[mm])
        r = l
    medoids.reverse()
    return np.array(medoids)


def _weighted_kmedoids(
    values: np.ndarray,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_init: int = 10,
    max_iter: int = 100,
) -> np.ndarray:
    """PAM-style k-medoids heuristic, operating directly on the weighted
    (value, weight) representation rather than literally replicating each
    value `weight` times -- this is an efficient equivalent of running
    kmedoids on the replicated multiset (exact for integer weights, see
    PORT_NOTES.md SS4.3's point 1), not an approximation of it.

    Ported as an option for MATLAB-output fidelity/validation (PORT_NOTES.md
    SS9 item 2); `_exact_weighted_kmedian` is the default and is what eq. 13
    actually asks for.
    """
    n = len(values)
    best_cost = np.inf
    best_medoid_idx = None

    def total_cost(medoid_idx):
        d = np.abs(values[:, None] - values[medoid_idx][None, :])
        nearest = np.argmin(d, axis=1)
        return np.sum(weights * d[np.arange(n), nearest]), nearest

    for _ in range(n_init):
        medoid_idx = rng.choice(n, size=min(k, n), replace=False)
        cost, _ = total_cost(medoid_idx)
        for _ in range(max_iter):
            improved = False
            for mi in range(len(medoid_idx)):
                current = medoid_idx[mi]
                for cand in range(n):
                    if cand in medoid_idx:
                        continue
                    trial = medoid_idx.copy()
                    trial[mi] = cand
                    trial_cost, _ = total_cost(trial)
                    if trial_cost < cost:
                        medoid_idx = trial
                        cost = trial_cost
                        improved = True
            if not improved:
                break
        if cost < best_cost:
            best_cost = cost
            best_medoid_idx = medoid_idx

    return values[best_medoid_idx]


def _legacy_minus50_labels(V_o: np.ndarray, N_o: np.ndarray, medoids: np.ndarray):
    """Faithful replication of the ORIGINAL MATLAB indexing (PORT_NOTES.md
    SS4.3's "-50 magic number"), without materializing the (potentially huge)
    replicated array.

    MATLAB builds V_o_1_weighted by replicating V_o[i] floor(N_o[i]) times,
    IN RAW STATE ORDER i=1:n^2 (NOT sorted by value) -- so state i's samples
    occupy 1-indexed positions (sum_no[i-1], sum_no[i]] where sum_no is the
    cumulative sum of floor(N_o) in raw state order. It then reads
    kmkm(sum_no[i]-50) as state i's label. Since every sample in a block has
    the identical value, that label is always nearest_medoid(V_o[j]) for
    whichever state j's block contains position sum_no[i]-50 -- which equals
    nearest_medoid(V_o[i]) itself only when floor(N_o[i]) > 50 (roughly).

    Returns (labels, crashed_states): labels[i] = -1 for any state where
    sum_no[i]-50 <= 0 (MATLAB would index with <=0 and error outright, not
    silently misassign) -- these are reported, not silently patched, but
    since we still need a complete ag_states to run decentralized training
    for the ablation, cluster_states() falls back to nearest-medoid-to-value
    for exactly these states (and only these) when method="legacy_minus50".
    """
    n2 = len(V_o)
    floor_N = np.floor(N_o).astype(np.int64)
    sum_no = np.cumsum(floor_N)  # sum_no[i] = total (raw-order) visits through state i inclusive

    labels = np.full(n2, -1, dtype=np.int64)
    crashed = []
    for i in range(n2):
        p = sum_no[i] - 50  # 1-indexed position into the (hypothetical) replicated array
        if p <= 0:
            crashed.append(i)
            continue
        j = int(np.searchsorted(sum_no, p, side="left"))  # state whose block contains position p
        labels[i] = np.argmin(np.abs(V_o[j] - medoids))
    return labels, crashed


def cluster_states(
    V_o: np.ndarray,
    N_o: np.ndarray,
    inf_bits: int,
    method: str = "kmedian",
    rng: np.random.Generator | None = None,
    return_diagnostics: bool = False,
):
    """Port of aggregate_states_SAIC.m (PORT_NOTES.md SS4.3).

    method="kmedian" (default) or "kmedoids": SS9 item 3's resolved fix --
    direct nearest-medoid-to-value assignment, exact, no low-visitation
    failure mode.

    method="legacy_minus50": faithful replication of the ORIGINAL MATLAB
    indexing bug (see _legacy_minus50_labels), for the ablation requested
    in chat -- comparing decentralized-training outcomes under the old,
    buggy labeling vs. the fixed one, holding the medoid-finding algorithm
    (kmedoids/PAM, matching original MATLAB) fixed. States where the bug
    would make MATLAB itself error out (index <= 0) fall back to
    nearest-medoid-to-value ONLY for those states, so a complete,
    runnable ag_states still comes out the other end -- flagged via
    return_diagnostics, not silently absorbed.

    Returns ag_states: (2**inf_bits, max_cluster_size) int array, 0-indexed
    raw states per cluster, padded with -1 -- NOT 0. MATLAB pads with 0
    because MATLAB states are 1-indexed (0 is never a real state); this
    port is 0-indexed, so state 0 IS real, and -1 is used as the "no state
    here" sentinel instead (PORT_NOTES.md SS2's index-mapping risk area).
    If return_diagnostics: also returns a dict with 'crashed_states' (list
    of raw states where the legacy index would have been <=0) and
    'misassigned_states' (states whose legacy label differs from the
    nearest-medoid-to-value label) -- both empty for method != "legacy_minus50".
    """
    k = 2**inf_bits
    diagnostics = {"crashed_states": [], "misassigned_states": []}

    if method == "kmedian":
        medoids = _exact_weighted_kmedian(V_o, N_o, k)
        cluster_id = np.argmin(np.abs(V_o[:, None] - medoids[None, :]), axis=1)
    elif method == "kmedoids":
        rng = rng or np.random.default_rng()
        medoids = _weighted_kmedoids(V_o, N_o, k, rng)
        cluster_id = np.argmin(np.abs(V_o[:, None] - medoids[None, :]), axis=1)
    elif method == "legacy_minus50":
        rng = rng or np.random.default_rng()
        medoids = _weighted_kmedoids(V_o, N_o, k, rng)
        legacy_labels, crashed = _legacy_minus50_labels(V_o, N_o, medoids)
        fixed_labels = np.argmin(np.abs(V_o[:, None] - medoids[None, :]), axis=1)
        cluster_id = np.where(legacy_labels < 0, fixed_labels, legacy_labels)
        diagnostics["crashed_states"] = crashed
        diagnostics["misassigned_states"] = [
            i for i in range(len(V_o)) if i not in crashed and legacy_labels[i] != fixed_labels[i]
        ]
    else:
        raise ValueError(
            f"Unknown clustering method {method!r}; expected 'kmedian', 'kmedoids', or 'legacy_minus50'"
        )

    counts = np.bincount(cluster_id, minlength=k)
    max_size = int(counts.max()) if len(counts) else 0
    ag_states = np.full((k, max_size), -1, dtype=np.int32)
    fill = np.zeros(k, dtype=np.int64)
    for state, cid in enumerate(cluster_id):
        ag_states[cid, fill[cid]] = state
        fill[cid] += 1

    if return_diagnostics:
        return ag_states, diagnostics
    return ag_states

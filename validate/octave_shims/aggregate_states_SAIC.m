function [ag_states_median] = aggregate_states_SAIC(qp_table,N_table_emerged,n,noa,inf_bits)
    % Octave reference copy matching jax_saic/clustering.py's resolved fix
    % (PORT_NOTES.md SS4.3/SS9 item 3), NOT the original MATLAB's `-50`
    % index trick. The original is provably buggy in a way that isn't just
    % a low-ns artifact: the goal cell's visitation count stays near the
    % 0.001 smoothing floor for ANY ns (episodes end immediately upon
    % arrival, confirmed both by static analysis and by triggering this
    % exact crash live under Octave -- kmkm(-32), a genuinely
    % out-of-bounds negative index, at ns=500). Since the whole point of
    % this reference run is to validate jax_saic's actual chosen behavior,
    % it uses the same fix here rather than reproducing a crash.
    [V_o_1,N_o_1] = sum_q_MultiAgent(qp_table,N_table_emerged,n,noa);

    k = 2^inf_bits;
    medoids = weighted_kmedian(V_o_1, N_o_1, k);

    d = abs(V_o_1(:) - medoids(:)');
    [~, cluster_id] = min(d, [], 2);  % 1..k

    % Pad to n^2 columns (the caller preallocates batch_ag_states_median
    % as (2^inf_bits, n^2, bn) -- SS4.3's shape-bug fix keeps the row count
    % correct (k=2^inf_bits, not the original's hardcoded 2) while matching
    % the caller's expected column count exactly, avoiding a separate
    % mismatch).
    ag_states_median = zeros(k, n^2);  % 1-indexed states here (Octave/MATLAB convention); 0 = pad
    fill = zeros(k,1);
    for state = 1:(n^2)
        cid = cluster_id(state);
        fill(cid) = fill(cid) + 1;
        ag_states_median(cid, fill(cid)) = state;
    end
end

function medoids = weighted_kmedian(values, weights, k)
    % Exact 1-D weighted k-median via DP (matches
    % jax_saic.clustering._exact_weighted_kmedian's algorithm).
    values = values(:);
    weights = weights(:);
    [sv, order] = sort(values);
    sw = weights(order);
    m = numel(sv);
    prefix_w = [0; cumsum(sw)];
    prefix_wv = [0; cumsum(sw .* sv)];

    dp = Inf(k+1, m+1);
    split = zeros(k+1, m+1);
    dp(1, 1) = 0;
    for j = 1:k
        for r = 1:m
            best = Inf; best_l = 0;
            for l = 0:(r-1)
                if dp(j, l+1) == Inf
                    continue
                end
                c = dp(j, l+1) + group_cost(sv, prefix_w, prefix_wv, l+1, r);
                if c < best
                    best = c; best_l = l;
                end
            end
            dp(j+1, r+1) = best;
            split(j+1, r+1) = best_l;
        end
    end

    medoids = zeros(k,1);
    r = m;
    for j = k:-1:1
        l = split(j+1, r+1);
        target = prefix_w(l+1) + (prefix_w(r+1) - prefix_w(l+1))/2;
        mm = l+1;
        while mm < r && prefix_w(mm+1) < target
            mm = mm + 1;
        end
        medoids(j) = sv(mm);
        r = l;
    end
end

function c = group_cost(sv, prefix_w, prefix_wv, l, r)
    % l, r are 1-indexed inclusive bounds into sv
    w_lr = prefix_w(r+1) - prefix_w(l);
    target = prefix_w(l) + w_lr/2;
    m = l;
    while m < r && prefix_w(m+1) < target
        m = m + 1;
    end
    median = sv(m);
    left = median*(prefix_w(m+1)-prefix_w(l)) - (prefix_wv(m+1)-prefix_wv(l));
    right = (prefix_wv(r+1)-prefix_wv(m+1)) - median*(prefix_w(r+1)-prefix_w(m+1));
    c = left + right;
end

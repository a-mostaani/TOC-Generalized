function [idx, C] = kmedoids(X, k, varargin)
  % Minimal MATLAB-Statistics-Toolbox-compatible kmedoids shim for Octave
  % (Octave-Forge's `statistics` package has never implemented kmedoids --
  % checked versions 1.4.3 through 1.8.4, none include it). PAM-style
  % (Partitioning Around Medoids) heuristic, matching MATLAB's default
  % algorithm family for kmedoids (not necessarily identical in tie-breaking
  % / initialization -- this is a best-effort, standard implementation, per
  % PORT_NOTES.md SS9 item 2's acknowledgment that exact kmedoids parity
  % isn't the validation bar).
  %
  % Only supports scalar (1-D) X and the 'Distance','euclidean' name-value
  % pair, since that's the only call pattern aggregate_states_SAIC.m uses.
  % [idx, C] = kmedoids(X, k, ...): idx is per-point cluster label (1..k),
  % C is the k medoid VALUES -- matching MATLAB's [idx, C] output order.

  X = X(:);
  n = numel(X);

  % Collapse to unique values + counts (weights) -- mathematically
  % equivalent to running PAM on the full (possibly heavily replicated)
  % array directly, since points with identical values always move
  % together and contribute additively to any candidate medoid's cost.
  % This is an internal efficiency choice only; the function's inputs and
  % outputs match calling kmedoids(X, k, 'Distance','euclidean') directly.
  [uvals, ~, inv_idx] = unique(X);
  counts = accumarray(inv_idx, 1);
  m = numel(uvals);
  k = min(k, m);

  best_cost = Inf;
  best_medoid_idx = [];
  n_init = 10;
  rand_state = rand('state');
  rand('state', 0);  % deterministic init, this is a reference run

  for init_i = 1:n_init
    medoid_idx = randperm(m, k);
    [cost, ~] = total_weighted_cost(uvals, counts, medoid_idx);
    improved = true;
    iter = 0;
    while improved && iter < 100
      improved = false;
      iter = iter + 1;
      for mi = 1:numel(medoid_idx)
        current_cost = cost;
        for cand = 1:m
          if any(medoid_idx == cand)
            continue
          end
          trial = medoid_idx;
          trial(mi) = cand;
          [trial_cost, ~] = total_weighted_cost(uvals, counts, trial);
          if trial_cost < current_cost
            medoid_idx = trial;
            current_cost = trial_cost;
            improved = true;
          end
        end
      end
      cost = current_cost;
    end
    if cost < best_cost
      best_cost = cost;
      best_medoid_idx = medoid_idx;
    end
  end

  rand('state', rand_state);

  [~, assign] = total_weighted_cost(uvals, counts, best_medoid_idx);
  C = uvals(best_medoid_idx);
  idx = C(assign);  % per-UNIQUE-value assignment, mapped back below
  idx = reshape(assign(inv_idx), size(X));  % per-ORIGINAL-point cluster label (1..k)
end

function [cost, assign] = total_weighted_cost(uvals, counts, medoid_idx)
  d = abs(uvals(:) - uvals(medoid_idx(:))');
  [mind, assign] = min(d, [], 2);
  cost = sum(counts .* mind);
end

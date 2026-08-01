function [v_o_1,N_o_1] = sum_q_MultiAgent(qp_table,N_table_coded,n,noa)
    % Octave-compatible copy of
    % "Fully Centralized - MultiAgent/sum_q_MultiAgent.m". Byte-for-byte
    % identical except ONE line: MATLAB's `sum(N_table, vecdim)` with a
    % multi-element `vecdim` sums over ALL listed dimensions at once;
    % this Octave build's `sum` does not implement that (verified with an
    % isolated test: sum(reshape(1:24,[2,3,4]), [1 3]) does not collapse
    % dimension 3 the way MATLAB does). Replaced with an explicit
    % sequential sum over each dimension in vecdim, which is algebraically
    % identical regardless of MATLAB/Octave differences (summing over a
    % set of axes one at a time equals summing over their union at once)
    % -- not an approximation, a provably-equal reformulation. This is the
    % only change in this file.
    N_table_coded = floor(N_table_coded);
    v_o_1=zeros(n^2,1);
    N_table=decod_N_table(N_table_coded,n,noa);
    prob_table = max(N_table,[],noa+1)/sum(sum(sum(N_table_coded(:,:)))); %it only counts the number of occurrance of optimal actions
                                     %since we are evaluating the ~optimal
                                     %policy,  other actions should not
                                     %occur.
    qp_table_new=decod_N_table(qp_table,n,noa);
    v_o_mat = max(qp_table_new,[],noa+1);    %it only keeps the value of the optimal action
                                         %since we are evaluating the ~optimal
                                         %policy,  other actions should not
                                         %occur.
    v_o_weighted = v_o_mat .* prob_table;
    N_o_1=zeros(n^2,1);
        %works for any number of agents also more than 3 agents
        inds(1:(noa-1))={':'};
        for i=1:n^2
            inds(noa) = {i};
            v_o_1(i)= sum(sum(v_o_weighted(inds{:})));
        end

    vecdim = 1:noa-1;
    vecdim = [vecdim,noa+1]; %the dimensions on which the next summation should be taken
    N_o_1 = N_table;
    for d = vecdim
        N_o_1 = sum(N_o_1, d);
    end
    N_o_1 = squeeze(N_o_1);
end


function N_table=decod_N_table(N_table_coded,n,noa) %here we just decode locations not actions,
%by decode we mean that after decoding the concatennated location we can
%draw the individual location of each agent
    n_dimensions = ones(1,noa)*n^2;
    n_dimensions = [n_dimensions,5^noa];
    N_table=zeros(n_dimensions);
    [w,~]=size(N_table_coded);
    if noa==2 %to have more than 3 agents this if statement should be generalized
        for i=1:w
            ps=ps_calc(i,n,noa);
            N_table(ps(1),ps(2),:) = N_table_coded(i,:);
        end
    elseif noa==3
        for i=1:w
            ps=ps_calc(i,n,noa);
            N_table(ps(1),ps(2),ps(3),:) = N_table_coded(i,:);
        end
    else
        warning("This code does not work for this many number of agents!");
    end
end

function ps=ps_calc(main_ps,n,noa)
ps=zeros(2,1);
counter = 0;
    for kk = noa:-1:1
        main_ps = main_ps - counter;
        ps(kk) = ceil((main_ps)/(n^2^(kk-1)))  ;
        counter = (ps(kk)-1)*(n^2^(kk-1));
    end
end

function main_ps = mps_calc(ps,n,noa)
    main_ps = 0;
    for kk = noa:-1:1
        main_ps = main_ps + (ps(kk)-1)*(n^2^(kk-1));
    end
    main_ps = main_ps +1;
end

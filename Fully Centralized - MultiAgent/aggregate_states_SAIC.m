function [ag_states_median] = aggregate_states_SAIC(qp_table,N_table_emerged,n,noa,inf_bits)
    [V_o_1,N_o_1] = sum_q_MultiAgent(qp_table,N_table_emerged,n,noa);


    %total number of time steps
    t_steps=floor(sum(N_o_1));
    V_o_1_weighted=zeros(t_steps,1);
    cnt=0;

    %creating V_o vector in which each value of each state is repeated
    %proportional to the probability of that state to ocurre
    for i=1:n^2
        V_o_1_weighted(cnt+1:cnt+floor(N_o_1(i)),1)=V_o_1(i)*ones(floor(N_o_1(i)),1);
        cnt=cnt+floor(N_o_1(i));
    end

    % State Aggregation by k-median clustering
    [kmkm,~]=kmedoids(V_o_1_weighted,2^inf_bits,'Distance','euclidean');

    sum_no=zeros(n^2,1);
    for i=1:n^2
        sum_no(i)=sum(floor(N_o_1(1:i)));
    end

    agr_st=zeros(n^2,1);
    for i=1:n^2
        agr_st(i)=kmkm(floor(sum_no(i))-50);
    end

    ag_states_median=zeros(2,n^2);
    cnt_i=0;
    for i=1:2^inf_bits
        cnt_i=1;
        for j=1:n^2
            if agr_st(j)==i
                ag_states_median(i,cnt_i)=j;
                cnt_i=cnt_i+1;
            end
        end
    end

end
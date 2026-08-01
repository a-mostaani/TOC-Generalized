function [v_o_1,N_o_1] = sum_q(qp_table,N_table_coded,n)
    v_o_1=zeros(n^2);
    N_table=decod_N_table(N_table_coded,n);
    N_o_1=zeros(n^2);
        for i=1:n^2 % for any possible observation of agent 1
            for j=1:n^2 %and any possible observation of agent 2
                main_ps=mps_calc([i,j],n);
                max_val=max(qp_table(main_ps,:));
                %probability of oj being equal to what it is right now (under currnt policy):
                prob_oj=sum(sum(N_table(:,j,:)))/sum(sum(sum(N_table_coded(:,:,:))));
                v_o_1(i)=v_o_1(i)+max_val*prob_oj;
            end
        end
    N_o_1=sum(sum(N_table,2),3);
end


function N_table=decod_N_table(N_table_coded,n) %here we just decode locations not actions,
%by decode we mean that after decoding the concatennated location we can
%draw the individual location of each agent
    N_table=zeros(n^2,n^2,25); 
    [w,~]=size(N_table_coded);
    for i=1:w
        ps=ps_calc(i,n);
        N_table(ps(1),ps(2),:) = N_table_coded(i,:);
    end
end

function ps=ps_calc(main_ps,n)
    ps=zeros(2,1);
    ps(1)=fix(main_ps/n^2)+ceil(rem(main_ps,n^2)/n^2);
    ps(2)=main_ps-(ps(1)-1)*n^2;
end

function main_ps = mps_calc(ps,n)
    main_ps= (ps(1)-1)*n^2+ps(2);
end

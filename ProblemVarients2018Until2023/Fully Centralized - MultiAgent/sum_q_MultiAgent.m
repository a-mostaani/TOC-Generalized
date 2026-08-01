function [v_o_1,N_o_1] = sum_q_MultiAgent(qp_table,N_table_coded,n,noa)
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
%     if noa==3


%         for i=1:n^2
%             v_o_1(i)= sum(sum(v_o_weighted(:,:,i)));
%         end
        %works for any number of agents also more than 3 agents
        inds(1:(noa-1))={':'};
        for i=1:n^2
            inds(noa) = {i};
            v_o_1(i)= sum(sum(v_o_weighted(inds{:})));
        end
        
%     elseif noa==2
%         for i=1:n^2 % for any possible observation of agent 1
%             for j=1:n^2 %and any possible observation of agent 2
%                 main_ps=mps_calc([i,j],n);
%                 max_val=max(qp_table(main_ps,:));
%                 %probability of oj being equal to what it is right now (under currnt policy):
%                 prob_oj=sum(sum(N_table(:,j,:)))/sum(sum(sum(N_table_coded(:,:,:))));
%                 v_o_1(i)=v_o_1(i)+max_val*prob_oj;
%             end
%         end
%     end
    
    vecdim = 1:noa-1;
    vecdim = [vecdim,noa+1]; %the dimensions on which the next summation should be taken
    N_o_1 = sum(N_table,vecdim);
%     N_o_1=sum(sum(sum(N_table,1),2),4);
%     N_o_1 = squeeze(N_o_1);
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
% ps(1)=fix(main_ps/n^2)+ceil(rem(main_ps,n^2)/n^2);
% ps(2)=main_ps-(ps(1)-1)*n^2;
counter = 0;
    for kk = noa:-1:1
        main_ps = main_ps - counter;
        ps(kk) = ceil((main_ps)/(n^2^(kk-1)))  ;
        counter = (ps(kk)-1)*(n^2^(kk-1));
    end
end

function main_ps = mps_calc(ps,n,noa)
%     if noa==2
%         main_ps = (ps(1)-1)*n^2+ps(2);
%     elseif noa==3
%         main_ps = (ps(1)-1)*n^(2*2)+ps(2) 
%     end
    main_ps = 0;
    for kk = noa:-1:1
        main_ps = main_ps + (ps(kk)-1)*(n^2^(kk-1));
    end
    main_ps = main_ps +1;
end

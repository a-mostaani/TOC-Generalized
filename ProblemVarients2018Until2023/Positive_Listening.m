%to run this script you need first to have centralized training to be done
%and its workspace to be there
% or you can load from the existing workspaces:
%load('prefectcom_n8_noa2_goalset22_allpar_plus_positivelistening');
%load('prefectcom_n8_noa2_goalset21_allpar_plus_positivelistening');

observ = zeros(2,1);
act    = zeros(2,1);


N_table_emerged = floor(N_table_emerged);



%% Q distribution computing:
q_max = max(max(qp_table));
q_levels = 0:q_max/99:q_max; %aggregating q-values to 100 values
q_count = zeros(100,1);

for i= 1:n^4
   [observ] = ps_calc(i,n);
   [val,arg] =  max(N_table_emerged(i,:));
   [act] = pa_calc(arg);
   q_st = qp_table(i,arg);
   q_count(ceil(q_st*10)) =  q_count(ceil(q_st*10)) + val;
end
q_prob = q_count/(sum(sum(q_count)));

%entropy
%removing zero probabilities
q_prob2=[];
counter = 0;
for i = 1:length(q_prob)
    if q_prob(i) ~= 0
        counter = counter + 1;
        q_prob2(counter,1)= q_prob(i);
    end
end

q_entropy = transpose(q_prob2)*(-log2(q_prob2));



%% Observation distribution computing
%agent 1's observations
observ_count = zeros(n^2,1);
for i= 1:n^4
    [observ] = ps_calc(i,n);
    observ_count(observ(1)) = observ_count(observ(1)) + sum(N_table_emerged(i,:));
end
observ_prob = observ_count/(sum(sum(observ_count)));
observ_prob(goal_set)=[];

%entropy
observ_entropy = transpose(observ_prob)*(-log2(observ_prob));



%% action distribution computing
%agent 2's actions
act_count = zeros(5,1);

for i= 1:n^4
   [observ] = ps_calc(i,n);
   [val,arg] =  max(N_table_emerged(i,:));
   [act] = pa_calc(arg);
   act_count(act(2)) =  act_count(act(2)) + val;
end
act_prob = act_count/(sum(sum(act_count)));

%entropy
act_entropy = transpose(act_prob)*(-log2(act_prob));



%% joint Q-Observation distribution computing

observ_q_count = zeros(n^2*100,1);
for i= 1:n^4
    [observ] = ps_calc(i,n);
    [val,arg] =  max(N_table_emerged(i,:));
    [act] = pa_calc(arg);
    q_st = qp_table(i,arg);
    observ_q_count(100*(observ(1)-1)+ceil(q_st*10)) = observ_q_count(100*(observ(1)-1)+ceil(q_st*10)) + sum(N_table_emerged(i,:));
end
observ_q_prob = observ_q_count/(sum(sum(observ_q_count)));

%entropy
%removing zero probabilities
observ_q_prob2=[];
counter = 0;
for i = 1:length(observ_q_prob)
    if observ_q_prob(i) ~= 0
        counter = counter + 1;
        observ_q_prob2(counter,1)= observ_q_prob(i);
    end
end

observ_q_entropy = transpose(observ_q_prob2)*(-log2(observ_q_prob2));
mi_observ_q = q_entropy + observ_entropy - observ_q_entropy;




%% joint act-observation distribution computing
observ_act_count = zeros(n^2*5,1);
for i= 1:n^4
    [observ] = ps_calc(i,n);
    [val,arg] =  max(N_table_emerged(i,:));
    [act] = pa_calc(arg);
    observ_act_count(5*(observ(1)-1)+act(2)) = observ_act_count(5*(observ(1)-1)+act(2)) + sum(N_table_emerged(i,arg));
end
observ_act_prob = observ_act_count / ( sum(sum(observ_act_count)) );

%entropy
%removing zero probabilities
observ_act_prob2 = zeros(length(find(observ_act_count~=0)),1);
counter = 0;

for i=1:length(observ_act_prob)
    if observ_act_prob(i)~=0
        counter = counter +1;
        observ_act_prob2(counter,1) = observ_act_prob(i);
    end
end

observ_act_entropy = transpose(observ_act_prob2)*(-log2(observ_act_prob2));
mi_observ_act = act_entropy + observ_entropy - observ_act_entropy;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=====================Measuring SAIC's Performance=========================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inf_bits = 1;

T = load('prefectcom_n8_noa2_goalset22_allpar_plus_positivelistening');
%load('agreggated_states_g22_n8_infbits1_median','ag_states_median');
%load('agreggated_states_g22_n8_infbits2_median','ag_states_median');
% load('agreggated_states_n3_g9_infbits3','ag_states');
% ag_states_median = ag_states;
%ag_states = [1 2 3 4 5 7 0 0; 6 8 0 0 0 0 0 0 ];  % n3g9bits1
%ag_states = [1 3 7 9 0 0 0 0; 2 4 6 8 0 0 0 0 ];  % n3g5bits1

%ag_states = [1 2 3 4 5 6 7 8 9 10 11 0 13 14 0 0; 15 12 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];  % n4g16bits1

% n8g22bits2:
%  ag_states = [1 2 25 26 33 34 41 42 43 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64;
%               3 4 5  8   9 10 11 17 18 27 35 36 40 44 45 46 47 48 0  0  0  0  0  0  0 ;
%               6 7 12 13 14 15 16 19 20 24 28 29 31 32 37 38 39 0  0  0  0  0  0  0  0 ;
%               23,21,30,0,0,0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0];

% n8g22bits1:
ag_states = [1  2  3  4 5 6 7 8 9 10 11 12 13 0 15 16 17 18 19 20 24 25 26 27 28 29 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64;
            21 23 30 14 0 0 0 0 0 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  ];



%% Communication distribution computing
%agent 1's observations
com_count = zeros(2^inf_bits,1);
for i= 1:n^4
    [observ] = ps_calc(i,n);
    comm = s_aggregate(observ,inf_bits,noa,ag_states);
    com_count(comm(1)) = com_count(comm(1)) + sum(N_table_emerged(i,:));
end
com_prob = com_count/(sum(sum(com_count)));
%observ_prob(goal_set)=[];

%entropy
com_entropy = transpose(com_prob)*(-log2(com_prob));


%% joint OptAct-communication distribution computing for SAIC
com_act_count = zeros(2^inf_bits*5,1);
for i= 1:n^4
    [observ] = ps_calc(i,n);
    comm = s_aggregate(observ,inf_bits,noa,ag_states);
    [~,arg] = max(T.qp_table(i,:));
    [act] = pa_calc(arg);
%     com_act_count(5*(comm(1)-1)+act(2)) = com_act_count(5*(comm(1)-1)+act(2)) + sum(N_table_emerged(i,arg));
    com_act_count(5*(comm(1)-1)+act(1)) = com_act_count(5*(comm(1)-1)+act(1)) + sum(N_table_emerged(i,arg));

end
com_act_prob = com_act_count / ( sum(sum(com_act_count)) );

%entropy
%removing zero probabilities
observ_act_prob2 = zeros(length(find(com_act_count~=0)),1);
counter = 0;

for i=1:length(com_act_prob)
    if com_act_prob(i)~=0
        counter = counter +1;
        observ_act_prob2(counter,1) = com_act_prob(i);
    end
end

com_act_entropy = transpose(com_act_prob)*(-log2(com_act_prob));
mi_com_act = act_entropy + com_entropy - com_act_entropy;








%% joint OptAct-communication distribution computing for SAIC - 2
%count joint occurance of comms and optimal actions
opt_action_communication_count = zeros(2^inf_bits*5,1);

for i = 1:n^4
    observation = ps_calc(i,n);
    SAIC_communication = s_aggregate(observation,inf_bits,noa,ag_states);
    [~,opt_action_ind] = max(qp_table(i,:));
    [obs_act_count,opt_action_ind_2] = max(N_table_emerged(i,:));
    if opt_action_ind_2 ~= opt_action_ind
        print('unexpected optimal action')
    end
    
    opt_action = pa_calc(opt_action_ind);
    act_com_ind = act_com_coder(opt_action,SAIC_communication,5^noa); %encoding two different scalars into one scalar
    opt_action_communication_count(act_com_ind(1)) = opt_action_communication_count(act_com_ind(1)) + obs_act_count;
end

opt_action_communication_prob = opt_action_communication_count/sum(sum(opt_action_communication_count)); 
opt_action_communication_entropy = transpose(opt_action_communication_prob)*(-log2(opt_action_communication_prob));
opt_action_communication_mi = act_entropy + com_entropy - opt_action_communication_entropy;









%% joint OptAct-communication distribution computing for SAIC - 3
%count joint occurance of comms and optimal actions
opt_action_communication_count = zeros(2^inf_bits*5^noa,1);

for i = 1:n^4
    observation = ps_calc(i,n);
    SAIC_communication = s_aggregate(observation,inf_bits,noa,ag_states);
    [~,opt_action_ind] = max(qp_table(i,:));
    [obs_act_count,opt_action_ind_2] = max(N_table_emerged(i,:));
    if opt_action_ind_2 ~= opt_action_ind
        print('unexpected optimal action')
    end
    
    opt_action = pa_calc(opt_action_ind);
    act_com_ind = act_com_coder(opt_action_ind,SAIC_communication, 5^noa); %encoding two different scalars into one scalar
    %obs_act_count: is the number of times that the *joint observation* and
    %the joint action have simultaneousely occured
    opt_action_communication_count(act_com_ind(1)) = opt_action_communication_count(act_com_ind(1)) + obs_act_count;
end

opt_action_communication_prob = opt_action_communication_count/sum(sum(opt_action_communication_count)); 
opt_action_communication_entropy = transpose(opt_action_communication_prob)*(-log2(opt_action_communication_prob));
opt_action_communication_mi = act_entropy + com_entropy - opt_action_communication_entropy;


















%% calculate main 
function pa=pa_calc(main_pa)
pa=zeros(2,1);
pa(1)=fix(main_pa/5)+ceil(rem(main_pa,5)/5);
pa(2)=main_pa-(pa(1)-1)*5;
end

function ps=ps_calc(main_ps,n)
ps=zeros(2,1);
ps(1)=fix(main_ps/n^2)+ceil(rem(main_ps,n^2)/n^2);
ps(2)=main_ps-(ps(1)-1)*n^2;
end

function main_ps = mps_calc(ps,n)
    main_ps= (ps(1)-1)*n^2+ps(2);
end

%state aggregation based on V_o computed by centralized algorithm
%be very careful to import the correct ag_states
function ag_ps=s_aggregate(ps,inf_bits,noa,ag_states)
    ag_ps=ones(noa,1);
    for j=1:noa
        for k=1:2^inf_bits            
            if any(ag_states(k,:)==ps(j))
                ag_ps(j)=k;
            end
        end
    end
end

function act_com_ind = act_com_coder(action,communication,n_action)
%n_action = is the cardinality of the action space
% in the case of rendezvous problem: the cardinality joint action space is
% 5^noa (noa = number of agents)

% in the case of rendezvous problem: the cardinality single agent's action 
% space is 5 
act_com_ind =  action + (communication-1)*n_action;
end




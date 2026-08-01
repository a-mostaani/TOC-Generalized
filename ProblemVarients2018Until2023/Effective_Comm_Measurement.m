%Measuring the efficiency of the communications

%to run this script you need first to have centralized training to be done
%and its workspace to be there
% or you can load from the existing workspaces:
load('prefectcom_n8_noa2_goalset22_allpar_plus_positivelistening');
%load('prefectcom_n8_noa2_goalset21_allpar_plus_positivelistening');

observ = zeros(2,1);
act    = zeros(2,1);
N_table_emerged = floor(N_table_emerged);
inf_bits = 1;
agent_n = 1; % which agent's action you want to analyse
agents =      1:noa;
agents_copy = 1:noa;
cross_act_com = 0; %measure mutual information of agent i's comm and agent j~=i's action if cross_act_com = 1
                   %measure mutual information of agent i's comm and agent i's    action if cross_act_com = 1
agents_copy(agent_n) = [];
cross_agent = agents_copy;

%ag_states = [1 2 3 4 5 7 0 0; 6 8 0 0 0 0 0 0 ];  % n3g9bits1
%ag_states = [1 3 7 9 0 0 0 0; 2 4 6 8 0 0 0 0 ];  % n3g5bits1

%ag_states = [1 2 3 4 5 6 7 8 9 10 11 0 13 14 0 0; 15 12 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];  % n4g16bits1


%% SAIC

% %n8g22bits3:
% load('agreggated_states_g22_n8_infbits3_median.mat', 'ag_states_median')
% ag_states = ag_states_median;



% n8g22bits2: (factual)
%  ag_states = [1 2 25 26 33 34 41 42 43 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64;
%               3 4 5  8   9 10 11 17 18 27 35 36 40 44 45 46 47 48 0  0  0  0  0  0  0 ;
%               6 7 12 13 14 15 16 19 20 24 28 29 31 32 37 38 39 0  0  0  0  0  0  0  0 ;
%               23,21,30,0,0,0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0];


% %n8g22bits1:
% load('agreggated_states_g22_n8_infbits1_median.mat', 'ag_states_median')
% ag_states = ag_states_median;


%% HOC:
% % n8g22bits1:
ag_states = [1  2  3  4 5 6 7 8 9 10 11 12 13 0 15 16 17 18 19 20 24 25 26 27 28 29 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64;
            21 23 30 14 0 0 0 0 0 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  ];





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









%% Jointly optimal action distribution
%count joint occurance of optimal actions given all possible joint
%observations
opt_jointaction_count = zeros(5^noa,1);

for i = 1:n^2
    %i is the observation of agent 1
        
    for j = 1:n^2
        temp_ind = mps_calc([i; j],n); %what is another joint observation in which the first agent has an identical observation
        [count,opt_joint_act_ind] = max(N_table_emerged(temp_ind,:)); % how many times o_1 jointly happened with other joint optimal actions
        %opt_joint_act_com_ind = act_com_coder(opt_joint_act_ind,SAIC_communication(1), 5^noa); %add to the count of the encoded joint communications and actions
        opt_jointaction_count(opt_joint_act_ind) = opt_jointaction_count(opt_joint_act_ind) + count; %count all the joint actions possible for a single o_1
    end
    
end

opt_jointaction_prob = opt_jointaction_count/sum(sum(opt_jointaction_count)); 


%removing zero probabilities
opt_jointaction_prob2 = zeros(length(find(opt_jointaction_count~=0)),1);
counter = 0;

for i=1:length(opt_jointaction_prob)
    if opt_jointaction_prob(i)~=0
        counter = counter +1;
        opt_jointaction_prob2(counter,1) = opt_jointaction_prob(i);
    end
end


opt_jointaction_entropy = transpose(opt_jointaction_prob2)*(-log2(opt_jointaction_prob2));






%% Single action distribution
%count joint occurance of optimal actions given all possible joint
%observations
opt_singaction_count = zeros(5,1);
%agent_n = 1; which agent's action you want to analyse

for i = 1:n^4
    %i is the joint observation of both agents
        [count,opt_joint_act_ind] = max(N_table_emerged(i,:)); % how many times o_1 jointly happened with other joint optimal actions
        action_vect = pa_calc(opt_joint_act_ind);
        if cross_act_com ==0
            single_action_ind = action_vect(agent_n);
        else
            single_action_ind = action_vect(cross_agent);
        end
        %opt_joint_act_com_ind = act_com_coder(opt_joint_act_ind,SAIC_communication(1), 5^noa); %add to the count of the encoded joint communications and actions
        opt_singaction_count(single_action_ind) = opt_singaction_count(single_action_ind) + count; %count all the joint actions possible for a single o_1
end

opt_singleaction_prob = opt_singaction_count/sum(sum(opt_singaction_count)); 


%removing zero probabilities
opt_singleaction_prob2 = zeros(length(find(opt_singaction_count~=0)),1);
counter = 0;

for i=1:length(opt_singleaction_prob)
    if opt_singleaction_prob(i)~=0
        counter = counter +1;
        opt_singleaction_prob2(counter,1) = opt_singleaction_prob(i);
    end
end


opt_singleaction_entropy = transpose(opt_singleaction_prob2)*(-log2(opt_singleaction_prob2));





%% Single action statistics
%count joint occurance of optimal actions given all possible joint
%observations
optimal_singleaction_count = zeros(5,1);
%agent_n = 1; which agent's action you want to analyse

for i=1:n^4
    [temp_count,joint_action_ind] = max(N_table_emerged(i,:));
    action_vector = pa_calc(joint_action_ind);
    if cross_act_com ==0
        action_scalar = action_vector(agent_n);
    else
        action_scalar = action_vector(cross_agent);
    end
    
    optimal_singleaction_count(action_scalar) = ...
    optimal_singleaction_count(action_scalar) + temp_count;
end

optimal_singleaction_prob = optimal_singleaction_count/sum(optimal_singleaction_count);

optimal_singleaction_entropy = transpose(optimal_singleaction_prob) * (-log2(optimal_singleaction_prob));






%% SAIC communication distribution
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





%% joint OptAct-communication distribution computing for SAIC - 3
%count joint occurance of comms and optimal actions
opt_jointaction_communication_count = zeros(2^inf_bits*5^noa,1);

for i = 1:n^2
    %i is the observation of agent 1
    
    SAIC_communication = s_aggregate(i,inf_bits,1,ag_states); %SAIC_communication is a vector containing two separate observations
    
    for j = 1:n^2
        temp_ind = mps_calc([i; j],n); %what is another joint observation in which the first agent has an identical observation
        [count,opt_joint_act_ind] = max(N_table_emerged(temp_ind,:)); % how many times o_1 jointly happened with other joint optimal actions
        opt_joint_act_com_ind = act_com_coder(opt_joint_act_ind,SAIC_communication(1), 5^noa); %add to the count of the encoded joint communications and actions
        opt_jointaction_communication_count(opt_joint_act_com_ind) = opt_jointaction_communication_count(opt_joint_act_com_ind) + count; %count all the joint actions possible for a single o_1
    end
    
end

opt_action_communication_prob = opt_jointaction_communication_count/sum(sum(opt_jointaction_communication_count)); 


%removing zero probabilities
opt_action_communication_prob2 = zeros(length(find(opt_jointaction_communication_count~=0)),1);
counter = 0;

for i=1:length(opt_action_communication_prob)
    if opt_action_communication_prob(i)~=0
        counter = counter +1;
        opt_action_communication_prob2(counter,1) = opt_action_communication_prob(i);
    end
end


opt_action_communication_entropy = transpose(opt_action_communication_prob2)*(-log2(opt_action_communication_prob2));
opt_action_communication_mi = opt_jointaction_entropy + com_entropy - opt_action_communication_entropy;






%% joint SingleAct-communication distribution computing for SAIC - 3
%count joint occurance of comms and optimal actions
opt_singleaction_communication_count = zeros(2^inf_bits*5,1);

for i = 1:n^4
    %i is the joint observation of agent 1 & 2
    observation_vect = ps_calc(i,n);
    observation_sc = observation_vect(agent_n); %observation scalar
    SAIC_communication = s_aggregate(observation_sc,inf_bits,1,ag_states); %SAIC_communication is a vector containing two separate observations
    
    [count,opt_joint_act_ind] = max(N_table_emerged(i,:)); % how many times o_1 jointly happened with other joint optimal actions
    action_vect = pa_calc(opt_joint_act_ind);
    if cross_act_com ==0
        single_action_ind = action_vect(agent_n);
    else
        single_action_ind = action_vect(cross_agent);
    end
    opt_single_act_com_ind = act_com_coder(single_action_ind,SAIC_communication(agent_n), 5); %add to the count of the encoded joint communications and actions
    
%     temp_ind = mps_calc([i; j],n); %what is another joint observation in which the first agent has an identical observation
%     [count,opt_joint_act_ind] = max(N_table_emerged(temp_ind,:)); % how many times o_1 jointly happened with other joint optimal actions
%     opt_joint_act_com_ind = act_com_coder(opt_joint_act_ind,SAIC_communication(1), 5^noa); %add to the count of the encoded joint communications and actions
    opt_singleaction_communication_count(opt_single_act_com_ind) = opt_singleaction_communication_count(opt_single_act_com_ind) + count; %count all the joint actions possible for a single o_1
%     
end

opt_singleaction_communication_prob = opt_singleaction_communication_count/sum(sum(opt_singleaction_communication_count)); 


%removing zero probabilities
opt_singleaction_communication_prob2 = zeros(length(find(opt_singleaction_communication_count~=0)),1);
counter = 0;

for i=1:length(opt_singleaction_communication_prob)
    if opt_singleaction_communication_prob(i)~=0
        counter = counter +1;
        opt_singleaction_communication_prob2(counter,1) = opt_singleaction_communication_prob(i);
    end
end


opt_singleaction_communication_entropy = transpose(opt_singleaction_communication_prob2)*(-log2(opt_singleaction_communication_prob2));
opt_singleaction_communication_mi = opt_singleaction_entropy + com_entropy - opt_singleaction_communication_entropy;






%% Joint Single action and communication statistics 
%count joint occurance of an agent's action with the communications
%received/sent (received: if cross_act_com=1, sent: if cross_act_com=0)
optimal_singleaction_communication_count = zeros(2^inf_bits*5,1);

for i=1:n^4
    [count_tmp,action_ind] = max(N_table_emerged(i,:));
    
    action_vector=pa_calc(action_ind);
    if cross_act_com ==0
        action_scalar = action_vector(agent_n);
    else
        action_scalar = action_vector(cross_agent);
    end
    
    
    observation_vector = ps_calc(i,n);
    observation_scalar = observation_vector(agent_n);
    
    communication_vector=s_aggregate(observation_vector,inf_bits,2,ag_states);
    communication_scalar = communication_vector(agent_n);
    
    encoded_act_com = act_com_coder(action_scalar,communication_scalar,5);
    
    optimal_singleaction_communication_count(encoded_act_com) = ...
    optimal_singleaction_communication_count(encoded_act_com) + count_tmp;    
    
end

optimal_singleaction_communication_prob = optimal_singleaction_communication_count/ ...
                                  sum(sum(optimal_singleaction_communication_count));
           
%removing zero probabilities
optimal_singleaction_communication_prob2 = zeros(length(find(optimal_singleaction_communication_count~=0)),1);
counter = 0;

for i=1:length(optimal_singleaction_communication_prob)
    if optimal_singleaction_communication_prob(i)~=0
        counter = counter +1;
        optimal_singleaction_communication_prob2(counter,1) = optimal_singleaction_communication_prob(i);
    end
end

optimal_singleaction_communication_entropy = transpose(optimal_singleaction_communication_prob2)*(-log2(optimal_singleaction_communication_prob2));
optimal_singleaction_communication_mi = opt_singleaction_entropy + com_entropy - optimal_singleaction_communication_entropy;





%% Joint Action vector and communication statistics
optimal_jointaction_communication_count = zeros(2^inf_bits*5^noa,1);

for i=1:n^4
    [count_tmp,action_ind] = max(N_table_emerged(i,:));
    
    observation_vector = ps_calc(i,n);
    observation_scalar = observation_vector(agent_n);
    
    communication_vector=s_aggregate(observation_vector,inf_bits,2,ag_states);
    communication_scalar = communication_vector(agent_n);
    
    act_com_ind = act_com_coder(action_ind,communication_scalar,5^noa);
    
    optimal_jointaction_communication_count(act_com_ind) = ...
    optimal_jointaction_communication_count(act_com_ind) + count_tmp;
end

optimal_jointaction_communication_prob = optimal_jointaction_communication_count/ ...
                                  sum(sum(optimal_jointaction_communication_count));
                              
                              
%removing zero probabilities
optimal_jointaction_communication_prob2 = zeros(length(find(optimal_jointaction_communication_count~=0)),1);
counter = 0;

for i=1:length(optimal_jointaction_communication_prob)
    if optimal_jointaction_communication_prob(i)~=0
        counter = counter +1;
        optimal_jointaction_communication_prob2(counter,1) = optimal_jointaction_communication_prob(i);
    end
end


optimal_jointaction_communication_entropy = transpose(optimal_jointaction_communication_prob2)*(-log2(optimal_jointaction_communication_prob2));
optimal_jointaction_communication_mi = opt_jointaction_entropy + com_entropy - optimal_jointaction_communication_entropy;







%% joint OptAct-observation distribution computing 
%count joint occurance of observations and optimal actions
opt_jointaction_observation_count = zeros(n^2*5^noa,1);

for i = 1:n^2
    %i is the observation of agent 1
        
    for j = 1:n^2
        joint_obsv = mps_calc([i; j],n); %what is another joint observation in which the first agent has observation = i
        [count,opt_joint_act_ind] = max(N_table_emerged(joint_obsv,:)); % how many times o_1 = i, o_2 = j jointly happened with other joint optimal actions
        opt_joint_act_obs_ind = act_com_coder(opt_joint_act_ind,i, 5^noa); %add to the count of the encoded joint obs and actions
        opt_jointaction_observation_count(opt_joint_act_obs_ind) = opt_jointaction_observation_count(opt_joint_act_obs_ind) + count; %count all the joint actions possible for a single o_1
    end
    
end

opt_action_observation_prob = opt_jointaction_observation_count/sum(sum(opt_jointaction_observation_count)); 


%removing zero probabilities
opt_action_observation_prob2 = zeros(length(find(opt_jointaction_observation_count~=0)),1);
counter = 0;

for i=1:length(opt_action_observation_prob)
    if opt_action_observation_prob(i)~=0
        counter = counter +1;
        opt_action_observation_prob2(counter,1) = opt_action_observation_prob(i);
    end
end


opt_action_observation_entropy = transpose(opt_action_observation_prob2)*(-log2(opt_action_observation_prob2));
opt_action_observation_mi = opt_jointaction_entropy + observ_entropy - opt_action_observation_entropy;





% f = fit(transpose([0.31 0.48 0.99 0.99 1]),transpose([0.0332    0.0523    0.4119    0.4119    0.4650]),'exp1')
% plot(f,transpose([0.31 0.48 0.99 0.99 1]),transpose([0.0332    0.0523    0.4119    0.4119    0.4650]))

% %Effective communications: (exact)

%figure
% %z(1,:) = [0.5521     0.6371    0.7143    0.7683];
%  z(1,:) = [0.07174    0.1014    0.1374    0.1702];
% %z(2,:) = [0.6757    0.7432    0.8108    0.8475];
%  z(2,:) = [0.1177    0.1547    0.2043    0.2372];
% %z(3,:) = [0.8205    0.8842    0.9363    0.9730];
%  z(3,:) = [0.2126    0.2754    0.341     0.3959];
% z(4,:) = [0.0332    0.0523    0.4119    0.4119];
% z(5,:) = [0.4650    0.4650    0.4650    0.4650];
% plot([2,3,4,5],transpose(z))
% legend("d=1","d=2","d=3","SAIC","HOC")


figure
 f = fit(transpose([0.31 0.48 0.99 0.99 1]),transpose([0.0332    0.0523    0.4119    0.4119    0.4650]),'exp1');
% the MI of "agent 1's communication" and "joint optimal action"
z(1,:) = f([0.5521     0.6371    0.7143    0.7683]);
z(2,:) = f([0.6757    0.7432    0.8108    0.8475]) ;
z(3,:) = f([0.8205    0.8842    0.9363    0.9730]) ;
z(4,:) =   [0.0332    0.0523    0.4119    0.4119]  ;
z(5,:) =   [0.4650    0.4650    0.4650    0.4650]  ;
plot([2,3,4,5],transpose(z))
legend("d=1","d=2","d=3","SAIC","HOC")






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
%the function works even if both action and communication inputs are
%vecotrs

%the function works even if the communication scalar/vector is anything
%else with any other meaning and cardinality of the space

%n_action = is the cardinality of the action space
% in the case of rendezvous problem: the cardinality joint action space is
% 5^noa (noa = number of agents)

% in the case of rendezvous problem: the cardinality single agent's action 
% space is 5 
act_com_ind =  action + (communication-1)*n_action;
end




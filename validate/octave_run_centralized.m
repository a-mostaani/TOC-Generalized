%% Reference run: centralized training + clustering (SAIC/ESAIC), for
%% comparison against jax_saic. Adapted from
%% "Fully Centralized - MultiAgent/benchmark_perfectcom_MultiAgent.m":
%% only the config header is edited (noa forced to 2 per ESAIC Theorem 1,
%% PORT_NOTES.md SS0.8; ns/bn parameterizable via pre-set workspace vars
%% instead of hardcoded literals; bn defaults to 1 per SS9 item 9) and the
%% display-only visualization + disconnected Lloyd's-algorithm block (not
%% ported, SS4.4/SS10) are dropped. The algorithm loop itself (lines
%% calling bench_policy_UCB / pbench_update / envir_gc / sum_q_MultiAgent /
%% aggregate_states_SAIC) is byte-for-byte unmodified from the original.

%close all
clc

%% Setup
n=3;
if ~exist('noa','var'); noa=2; end        % SS0.8: ESAIC centralized phase always runs at noa=2
if ~exist('ns','var'); ns=4000; end       % episodes; MATLAB comment default for noa=2,n=3 is 120000
if ~exist('bn','var'); bn=1; end          % SS9 item 9: one batch suffices
if ~exist('out_file','var'); out_file='central_reference.mat'; end

end_learn=0.850;

goal_set=9;

inf_bits=2;
%windy_envir        
                    %if you need a windy environment, you can choos windy_envir
                    %function instead of envir

%variables that can be modified in cdedcie():
%alpha=0.2; 
%sweep=0; %if sweep is one, we update the whole q_table at the time that the
         % function cdecide is called
         %if sweep is off, then only update the q_table for the current
         %state of the agents
%gamma=0.9;

best_rew=10;


                    
tau_k=0.005;        
                    %the constant value based on which tau will be updated in each
                    %new episode
gamma=.9;
%% Zero initialization
batch_rew=zeros(ns,bn);
batch_counter=zeros(ns,bn);
batch_qp_table=zeros(n^(2*noa),5^noa,bn);
batch_N_table_emerged=zeros(n^(2*noa),5^noa,bn);

s_space=1:1:n^2;
s_space(goal_set)=[];




%%%%%%%%%%%%%%%%%%%%%%%% Timer begins:
elapsed_time_accumul = zeros(bn,1);
tic



for b=1:bn 
        disp(b)
        wind=zeros(1,2);
                            %if environment is windy, it can get 0 or 1 in x or y
                            %direction
        %wind_loc           
                            %please note that if you want to change the setting
                            %related to the wind_loc you should do it in the
                            %initialization part of envir_windy

                    
        %ca=zeros(noa,1);      
                            %communication action of each of agents (each row)
                            %it can be specified based on communcation policy, there are
                            %two policies studied in this paper:
                            %1-sending the current position
                            %2-learning how to communicate using one bit of data
                    
        %policy             
                            %if you want to change the policy you should change it
                            %in cdecide or pdecide functions
        %cs=ones(noa,noa-1);      
                            %communication state of each of agents (each row)
                            %This is equal to communication action of the other agents in
                            %the previous step
        main_pa=randi(5^noa);
        pa=pa_calc(main_pa,noa);
                            %position action of each of agents (each row)
                            %done based on RL 
        ps_ind=randi(n^2-1,noa,1);
        ps=zeros(noa,1);
        for kk=1:noa
            ps(kk) = s_space(ps_ind(kk));
        end
        
        main_ps = 0;
        for kk=1:noa
            if kk>1
                main_ps = main_ps+ (ps(kk)-1)*(n^2)^(kk-1);
            else
                main_ps = main_ps+ ps(kk);
            end
            
        end
        %main_ps=(ps(1)-1)*n^2+ps(2);
        
                            %position state of each of agent (each row)
                            %based on pa, environment will do the calculations to
                            %determine the nex position state
                            %at initialization, this value is determinde randomly
                            %but can't be the "terminal state"
        ter=0;
                            %indicates if the terminal state has been achieved or
                            %not and if it has been achieved how many agents has
                            %been into it. e.g. ter=0 : not achived
                                               %ter=1 : achived, one agent on it
                                               %ter=2 : achived, two agent on it
                                               %ter=n : achived, n agent on it
        rew=zeros(ns,1);    
                            %general rward for each episode simulation
        temp_rew=0;
                            %this value is used inside while loop to be transferred
                            %to cdecide and pdecide when a reward has been achieved
                            %this would let the q function for that state action
                            %being updated

        counter=zeros(ns,1);
                            %number of steps taken in each episode simulaton
                    

        %last_ps=zeros(noa,1); %last position state of each agent
        %last_cs=zeros(noa,1); %last communication state of each agent

        cumul_rew=0;

        %qc3_table=0.5*ones(n^2,2);
        %qc3_table(n^2,:)=[2,2];
                            %the q table of communication actions for scenario 3
        qp_table=0.02*ones((n^2)^noa,5^noa);
        N_table=0.001*ones((n^2)^noa,5^noa);
        N_table_emerged=0.001*ones((n^2)^noa,5^noa);
        %qp3_table(n^2,:,:)=2*ones(2,5);
                            %the q table of position actions for sceneario 3
                    
        %load('qc3_table.mat','qc3_table')
        %load('qp3_table.mat','qp3_table')

        %saved_qc_8=zeros(ns,2);
                                %saving the 8th row of qc table throughout time to
                                %to see how it evolves
        %saved_qc_6=zeros(ns,2);
        %saved_qp_8=zeros(ns,5);
                                %saving the 8th row of qp table throughout time to
                                %to see how it evolves
        %saved_qp_6=zeros(ns,5);

        %% Episode iteration
        
        for i=1:ns
            %Episode initialization
            disp(i)
            temp_rew=0;
% % %             rew_winner=[];
            counter(i)=1;

            % update tau, but above step 40'000 matlab will be unable to handle the
            % very big numbers so we don't go beyond...
% %             if i<=20000
% %                 tau=1/(1+i*tau_k);
% %             else
% %                 tau=1/(1+20000*tau_k); 
% %             end
    
            %random initialization of position states and actions (MultiAgent verion)
            ps_ind=randi(n^2-1,noa,1);
            ps=zeros(noa,1);
            for kk=1:noa
                ps(kk) = s_space(ps_ind(kk));
            end

            main_ps = 0;
            for kk=1:noa
                if kk>1
                    main_ps = main_ps+ (ps(kk)-1)*(n^2)^(kk-1);
                else
                    main_ps = main_ps+ ps(kk);
                end

            end
            
            main_pa=randi(5^noa);
            pa=pa_calc(main_pa,noa);
 
            while 3==3    
                        % why 3==3 ?
                        % while loop could not be conditioned on terminal state
                        % because after wev'e got to the terminal state still the 
                        % table updates should be done.
                        % Instead, at the end of each while loop it is checked if
                        % we've got to the terminal state or no
    
 

                if counter(i)~=1   %%%%make sure if the if statement is necessary
                    %updating position table:
                    [qp_table] = pbench_update(main_ps,last_ps,main_pa,temp_rew,qp_table); 
                end
        
                %SELECT POSITION ACTION AND UPDATE POSITON STATE
                %select position action 
                ucb_counter=sum(counter(1:i));
                [main_pa] = bench_policy_UCB(main_ps,0.005,qp_table,N_table,ns,i,ucb_counter,end_learn,best_rew,noa);
                pa=pa_calc(main_pa,noa);
                
                %Update UCB counter
                N_table(main_ps,main_pa)=N_table(main_ps,main_pa)+1;
                
                %Update UCB_emerged counter
                if i>end_learn*ns
                    N_table_emerged(main_ps,main_pa)=N_table_emerged(main_ps,main_pa)+1;
                end
                
                
                %canceling action if the agent is in the terminal state
                ps_check=ps_calc(main_ps,n,noa);
                if sum(ps==goal_set)>=1
                    main_pa=5^noa;
                end
                
                                     
                %saving the previous position state before it is updated
                last_ps=main_ps;
                                       
                %OBTAIN NEW ENVIRONMENT STATES AND REWARD
                [ps,err,ter]=envir_gc(ps,pa,n,noa,goal_set); 
                main_ps = 0;
                for kk=1:noa
                    if kk>1
                        main_ps = main_ps+ (ps(kk)-1)*(n^2)^(kk-1);
                    else
                        main_ps = main_ps+ ps(kk);
                    end

                end

                
                
                if ter==noa 
                                                  %calculating temp_reward
                                                  %this figure will be used to
                                                  %update q functions
                                          
                    temp_rew=best_rew;%*gamma^(counter(i));
                    rew_winner=[];
                    for ii=1:noa
                        if ps(ii)==n^2
                            rew_winner=[rew_winner,ii];
                        end  
                    end
                elseif ter>=1
                    temp_rew=1;%*gamma^(counter(i));
                    rew_winner=[];
                    for ii=1:noa
                        if ps(ii)==n^2
                            rew_winner=[rew_winner,ii];
                        end  
                    end
                end
        
        
        
        
                %UPDATING TABLES:

                %counting the number of steps in the current episode
                counter(i)=counter(i)+1;

        
                %CHECKING WHILE LOOP CONDITION
                if ter >=1 
                    if i<end_learn*ns    
                                                  %in order to make sure that each loop is completed before termination
                                                  %while loop condition is always
                                                  %active but we check the
                                                  %condition at the end of each
                                                  %loop
                                          
                                                  %The reason is that we want to make sure that the q fucntions are updated even if we are in the terminal state
                        %updating position table:
                        [qp_table] = pbench_update(main_ps,last_ps,main_pa,temp_rew,qp_table);                   
                        break
                    else
                        break
                    end
                end
            end

        %episode summerize
            if temp_rew>1
                rew(i)=best_rew*gamma^(counter(i)-1);
            else
                rew(i)=1*gamma^(counter(i)-1);
            end
            cumul_rew=cumul_rew+rew(i);
% % %             saved_qc_8(i,:)=qc3_table(n^2-1,:);
% % %             saved_qc_6(i,:)=qc3_table(n^2-n,:);
% % %             saved_qp_8(i,:)=qp_table(n^2-1,1,:);
% % %             saved_qp_6(i,:)=qp_table(n^2-n,1,:);
        end
        %save('qc3_table.mat','qc3_table')
        %save('qp3_table.mat','qp3_table')

        
      
%BATCH SAVING        
batch_rew(:,b)=rew;
batch_counter(:,b)=counter;
batch_qp_table(:,:,b)= qp_table;
batch_N_table_emerged(:,:,b)= N_table_emerged;
elapsed_time_accumul(bn) = toc;
end



elapsed_time = toc
%%%%%%%%%%%%%%% Timer ends



%% State Aggregation for Information Compression - SAIC:


inf_bits=2;
%syms = 5;

% %     [V_o_1,N_o_1] = sum_q_MultiAgent(qp_table,N_table_emerged,n,noa);
% % 
% % 
% %     %total number of time steps
% %     t_steps=floor(sum(N_o_1));
% %     V_o_1_weighted=zeros(t_steps,1);
% %     cnt=0;
% % 
% %     %creating V_o vector in which each value of each state is repeated
% %     %proportional to the probability of that state to ocurre
% %     for i=1:n^2
% %         V_o_1_weighted(cnt+1:cnt+floor(N_o_1(i)),1)=V_o_1(i)*ones(floor(N_o_1(i)),1);
% %         cnt=cnt+floor(N_o_1(i));
% %     end
% % 
% %     % State Aggregation by k-median clustering
% %     [kmkm,~]=kmedoids(V_o_1_weighted,2^inf_bits,'Distance','euclidean');
% % 
% %     sum_no=zeros(n^2,1);
% %     for i=1:n^2
% %         sum_no(i)=sum(floor(N_o_1(1:i)));
% %     end
% % 
% %     agr_st=zeros(n^2,1);
% %     for i=1:n^2
% %         agr_st(i)=kmkm(floor(sum_no(i))-50);
% %     end
% % 
% %     ag_states_median=zeros(2,n^2);
% %     cnt_i=0;
% %     for i=1:2^inf_bits
% %         cnt_i=1;
% %         for j=1:n^2
% %             if agr_st(j)==i
% %                 ag_states_median(i,cnt_i)=j;
% %                 cnt_i=cnt_i+1;
% %             end
% %         end
% %     end
% % 
% %     %save('agreggated_states_g64_infbits2', 'ag_states_median') 

batch_ag_states_median = zeros(2^inf_bits,n^2,bn);
for b=1:bn
    [batch_ag_states_median(:,:,b)] = aggregate_states_SAIC(batch_qp_table(:,:,b),batch_N_table_emerged(:,:,b),n,noa,inf_bits);
end

%% Save results for comparison with jax_saic
qp_table = batch_qp_table(:,:,1);
N_table_emerged = batch_N_table_emerged(:,:,1);
ag_states_median = batch_ag_states_median(:,:,1);
save('-v7', out_file, 'qp_table', 'N_table_emerged', 'ag_states_median', 'n', 'noa', 'ns', 'inf_bits', 'best_rew', 'gamma', 'end_learn');
printf('Saved centralized-training results to %s\n', out_file);

% EoC_SAIC_3Agents.m (unmodified) hardcodes
% load('agreggated_states_n3_g9_infbits2_realSAIC','batch_ag_states_median')
% -- rather than editing that function, supply exactly the filename/variable
% it expects (a 3-D array, singleton batch dim, matching the ORIGINAL
% authors' own workflow of save-then-load between the two phases).
if exist('ag_states_out_file', 'var')
  batch_ag_states_median = reshape(ag_states_median, size(ag_states_median,1), size(ag_states_median,2), 1);
  save('-v7', ag_states_out_file, 'batch_ag_states_median');
  printf('Saved batch_ag_states_median to %s\n', ag_states_out_file);
end

% pa_calc / ps_calc / mps_calc: moved to validate/octave_shims/ as standalone
% function files -- this Octave build doesn't support MATLAB-style local
% functions at the end of a script file (confirmed with an isolated test).

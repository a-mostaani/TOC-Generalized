function [rew,qp_table_main,qc_table,counter,NE_table_emerged]=EoC_SAIC_ActorCritic_3Agents(scen,n,noa,ns,bits,inf_bits,best_rew,worst_rew,goal_set,gamma,tau_k,ca,cs,pa,ps,ter,rew,temp_rew,counter,qc_table,qp_table_main,bsc_p,end_learn,update_tables)

training = update_tables;
%load('agreggated_states_g22_n8_infbits1_median','ag_states_median');
%load('agreggated_states_g22_n8_infbits2_median','ag_states_median');
% load('agreggated_states_n3_g9_infbits3','ag_states');
% ag_states_median = ag_states;

%ag_states = [1 2 3 4 5 7 0 0; 6 8 0 0 0 0 0 0 ];  % n3g9bits1
ag_states = [1 2 3 4 5 6 7 8 9 10 11 0 13 14 0 0; 15 12 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];  % n4g16bits1

% n8g22bits2:
%  ag_states = [1 2 25 26 33 34 41 42 43 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64;
%               3 4 5  8   9 10 11 17 18 27 35 36 40 44 45 46 47 48 0  0  0  0  0  0  0 ;
%               6 7 12 13 14 15 16 19 20 24 28 29 31 32 37 38 39 0  0  0  0  0  0  0  0 ;
%               23,21,30,0,0,0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0];

% n8g22bits1:
%ag_states = [1  2  3  4 5 6 7 8 9 10 11 12 13 0 15 16 17 18 19 20 24 25 26 27 28 29 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64;
%             21 23 30 14 0 0 0 0 0 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  ];

%load('qp_table_n3_g9_noa3_3.mat','qp_table')
%load('qp_table_n3g9noa3_1.mat','qp_table_main')

%load('qp_table_n4_g16_noa3_3_5p8.mat','qp_table')
load('qp_table_n4g16noa3_2.mat','qp_table_main')

if training ==1
    qp_table_central = qp_table;
end

N_table=0.001*ones((n^2)^noa,5^noa);


%ag_states = [1 3 7 9 0 0 0 0; 2 4 6 8 0 0 0 0 ];  % n3g5bits1


ag_states_median = ag_states;
term_hist = zeros(ns,1);
monit = zeros(1,ns);
monit_count = 0;
exp_rew = zeros(ns,1);
%ag_states=ag_states_median;
%% Numerical Simulator of Emergence of communication among rl agents under coordination environment
%Started: 11/02/2022
%Functions called : envir(ps,pa,n,noa) / envir_windy(ps,pa,n,noa,windy) , pdecide(ps,last_ps,cs,last_cs,scen,pa,temp_rew,rew_winner,tau), cdecide(ps,last_ps,ca,scen,temp_rew,rew_winner,tau)


%use new functions of cpolicy cosumized cupdate customized and ...
%goal_set=[n*n];

NE_table=0.001*ones(noa,n^2,2^((noa-1)*inf_bits),5);   %check if works for >2 agents : you are assuming the receipt of one comm message
%NC_table=0.001*ones(2,n^2,2^bits);
NE_table_emerged=0.001*ones(noa,n^2,2^((noa-1)*inf_bits),5);   %check if works for >2 agents: you are assuming the receipt of one comm message

%% Setup
stuck_counter=0;
%inf_bits=length(de2bi(n*n-length(goal_set)));
%inf_bits=3;
cs=ones(noa,noa-1,inf_bits);  %only for position_based signalling scenario    %check if works for >2 agents
legit_initial_pos = 1:1:n^2; %all locations
legit_initial_pos(goal_set) = []; %removing the terminal location
bits_sent_per_agent = inf_bits;
        %% Episode iteration
        for i=1:ns
            %Episode initialization
            disp(i)
            %waitbar(((b-1)/en)+i/(ns*b),h)
            temp_rew=0;
            ter=0;
            rew_winner=[];
            counter(i)=1;

            % update tau, but above step 40'000 matlab will be unable to handle the
            % very big numbers so we don't go beyond...
            %tau=0.0001;

            if i<=40000
                tau=1/(1+i*tau_k);
            else
                tau=1/(1+40000*tau_k); 
            end

    
            %random initialization of position states and actions from
            %non-terminal locations
            ps_ind=randi(n*n-1,noa,1);        %generating a random index of position
            ps    =legit_initial_pos(ps_ind);  %generating a random non-terminal position
            
            %computed the corresponding aggregated ps
            ag_ps=s_aggregate(ps,inf_bits,noa,ag_states);
            
            pa=randi(5,noa,1);
    
            %random initialization of communication action ca, it depens on the scenario
            %selected
            switch scen
                case 1
                    ca=randi(2,noa,1);                             
                case 2
                    ca=ps;
                case 3
                    ca=randi(2,noa,bits)-ones(noa,bits);   %check if works for >2 agents: ca is generated regardless of the ag_ps
            end
    
        %episode run    
            while 3==3    
                        % why 3==3 ?
                        % while loop could not be conditioned on terminal state
                        % because after wev'e got to the terminal state still the 
                        % table updates should be done.
                        % Instead, at the end of each while loop it is checked if
                        % we've got to the terminal state or no
    
                ucb_counter=sum(counter(1:i));
                %SELECT COMM. ACTION AND UPDATE COMM. STATE
                %select communication action
                if bits>inf_bits

                        [ca] = de2bi(ag_ps-ones(noa,1),inf_bits);
        % %                     gen=[1 0 0 1 0 1
        % %                          0 1 0 1 1 1
        % %                          0 0 1 0 1 1];
                        encData = encode(ca,bits,inf_bits,'cyclic/fmt');                                                

                elseif bits==inf_bits
                        [ca] = de2bi(ag_ps-ones(noa,1),inf_bits);

                elseif bits<inf_bits
                        error('The number of bits selected is not enough to send the position information')
                        disp('The minimum number of bits which will not have any redundency is:')
                        length(de2bi(n*n-1,bits))
                end
                                             
        
                %saving previouse communication state before it is updated
                last_cs=cs;
                
                if noa > 2 %join all the received communications by one agent in one single vector:
                    last_cs_new=zeros(noa,(noa-1)*bits);
%                     for kk=1:noa %for multi agent you should extend this expression (works only for 3 agents)
%                         last_cs_new(kk,:)=[squeeze(last_cs(kk,1,:)); squeeze(last_cs(kk,2,:))];
%                     end
                     bit_counter = 0;
                     for kk=1:noa 
                         bit_counter = 0;
                         for jj = 1:noa-1
                             last_cs_new(kk,bit_counter + 1 : bit_counter + bits_sent_per_agent)=squeeze(last_cs(kk,jj,:));
                             bit_counter = bit_counter + bits_sent_per_agent;  %does not work for heterogeneous bit rates
                         end
                     end


                end
                % Use last_cs_new and cs_new in the multi-agent cases with
                % more than two agents
                
                
                %communication state update:
                %the last ca of the other agent(s) is the cs of the current
                %agent in case that no noise is added. Otherwise, noise
                %would change the signlas sent through the communication
                %channel.
                if bits>inf_bits
                        %communication state update:
                        %the previous ca of the other agent is the cs of the current
                        %agent in case that no noise is added. Otherwise, noise
                        %would change the signlas sent through the communication
                        %channel.


                        for j=1:noa
                            cs_temp=encData;
                            cs_temp(j,:)=[];
                            cs_bc(j,:,:)=cs_temp;
                        end
                        %Sending cs_beforechannel to channel 
                        rec=bsc_ch(cs_bc,bsc_p);

                        for k=1:noa
                            cs(k,:) = decode(rec(k,:),bits,inf_bits,'cyclic/fmt');
                        end
                elseif bits==inf_bits
                    %rate-limited reliable communication channel:
                    for k=1:noa
                        ca_temp = ca;
                        ca_temp(k,:) = [];
                        cs(k,:,:) = ca_temp;
                    end  
                    
                    if noa > 2 %join all the received communications by one agent in one single vector:
                        cs_new=zeros(noa,(noa-1)*bits);
%                         for kk=1:noa %for multi agent you should extend this expression (works only for 3 agents)
%                             cs_new(kk,:)=[squeeze(cs(kk,1,:)); squeeze(cs(kk,2,:))];
%                         end
                         bit_counter = 0;
                         for kk=1:noa 
                             bit_counter = 0;
                             for jj = 1:noa-1
                                 cs_new(kk,bit_counter + 1 : bit_counter + bits_sent_per_agent)=squeeze(cs(kk,jj,:));
                                 bit_counter = bit_counter + bits_sent_per_agent;  %does not work for heterogeneous bit rates
                             end
                         end
                    end
                    
                elseif bits<inf_bits
                        error('The number of bits selected is not enough to send the position information')
                        disp('The minimum number of bits which will not have any redundency is:')
                        length(de2bi(n*n-1,bits))
                end
                 
                

                if counter(i)~=1  && update_tables==1 
                    %updating position table: %check if works for >2
                    %agents: q_tables should consider two comm messages
                    [qp_table_main] = pupdate_customized_nbits_ma(ps,last_ps,cs,last_cs,scen,pa,temp_rew,rew_winner,tau,qp_table_main); 
                
                    %CHECKING WHILE LOOP CONDITION
                    if ter >=1 
                                                      %in order to make sure that each loop is completed before termination
                                                      %while loop condition is always
                                                      %active but we check the
                                                      %condition at the end of each
                                                      %loop

                                                      %The reason is that we want to make sure that the q fucntions are updated even if we are in the terminal state
                        %updating position table:
                            term_hist(i)=ter;
                        break
                    end
                    
                    if (bi2de(cs_new(3,:))+1==4) & (ps(3) ==14)
                       monit_count = monit_count+1;
                       monit(monit_count) = qp_table_main(3,14,4,3); 
                    end
                    
                elseif ter >=1
                       term_hist(i)=ter;
                       break
                end
        
                %SELECT POSITION ACTION AND UPDATE POSITON STATE
                %select position action %check if works for >2 agents :
                %qp_table should be updated to capture the value of agent's
                %observation together with two comm messages received
                policy = 'greedy';
                if training == 0
                    [pa] = ppolicy_customized_nbits_UCB_bestrew_ma(ps,cs,scen,tau,qp_table_main,NE_table,ucb_counter,ns,end_learn,best_rew,policy);
                                                      %(ps,cs,scen,tau,qp_table,NE_table,ucb_counter,ns,end_learn)
                    %Critic Selects the action
                elseif training == 1
                    main_ps = ps_calc(ps,n,noa);
                    [main_pa] = bench_policy_UCB(main_ps,0.005,qp_table_central,N_table,ns,i,ucb_counter,end_learn,best_rew,noa,policy);
                    pa=pa_calc(main_pa,noa);
                end
                %Update UCBE counter
                %condition to check: sum(ag_ps == 2)==3 & sum(envir_gc(ps,pa,n,noa,goal_set) ==22)==3
                if noa>2
                    for k=1:noa
                        NE_table(k,ps(k),bi2de(cs_new(k,:))+1,pa(k))=NE_table(k,ps(k),bi2de(cs_new(k,:))+1,pa(k))+1;
                    end
                else
                    for k=1:noa %check if works for >2 agents
                        NE_table(k,ps(k),bi2de(transpose(squeeze(cs(k,1,:))))+1,pa(k))=NE_table(k,ps(k),bi2de(transpose(squeeze(cs(k,1,:))))+1,pa(k))+1;
                    end
                end
                
                %Update UCB_emerged counter
                if i>end_learn*ns
                    if noa>2
                        for k=1:2
                            NE_table_emerged(k,ps(k),bi2de(cs_new(k,:))+1,pa(k))=NE_table(k,ps(k),bi2de(cs_new(k,:))+1,pa(k))+1;
                        end
                    else
                        for k=1:2
                            NE_table_emerged(k,ps(k),bi2de(transpose(squeeze(cs(k,1,:))))+1,pa(k))=NE_table(k,ps(k),bi2de(transpose(squeeze(cs(k,1,:))))+1,pa(k))+1;
                        end
                    end

                end
                
                
                %canceling action if the agent is in the terminal state
                for j=1:noa
                    if ps(j)== [goal_set]
                        pa(j)=5;
                    end
                end
                                     
                %saving the previous position state before it is updated
                last_ps=ps;
                                       
                %environment, position state update
                [ps,~,ter]=envir_gc(ps,pa,n,noa,goal_set); 
                                                  %envir is a function which models the environment
                                                  %it tells if terminal state has
                                                  %been achieved, it tells if any
                                                  %error has occured
                                
                                          
                ag_ps=s_aggregate(ps,inf_bits,noa,ag_states);
                % Don't get stuck
                if length(find(last_ps==ps))==noa
                    stuck_counter=stuck_counter+1;
                    if stuck_counter==10
                        temp_rew=0;
                        if update_tables==1
% %                             [qc_table] = cupdate_customized_nbits_ma(ps,last_ps,ca,scen,temp_rew,rew_winner,qc_table);
% %                             [qp_table] = pupdate_customized_nbits_ma(ps,last_ps,cs,cs,scen,pa,temp_rew,rew_winner,tau,qp_table);
                                                last_cs_new=zeros(noa,(noa-1)*bits);
%                     for kk=1:noa %for multi agent you should extend this expression (works only for 3 agents)
%                         last_cs_new(kk,:)=[squeeze(last_cs(kk,1,:)); squeeze(last_cs(kk,2,:))];
%                     end
                            last_cs_new=zeros(noa,(noa-1)*bits);
                            bit_counter = 0;
                            for kk=1:noa
                                bit_counter = 0;
                                for jj = 1:noa-1
                                    last_cs_new(kk,bit_counter + 1 : bit_counter + bits_sent_per_agent)=squeeze(last_cs(kk,jj,:));
                                    bit_counter = bit_counter + bits_sent_per_agent;  %does not work for heterogeneous bit rates
                                end
                            end
                            for k=1:noa
                                %the following line was commented out
                                %because that is enough to change qp_table
                                %to not get stuck any more
                                %qc_table(k,last_ps(k),bi2de(ca(k,:))+1)=median(qc_table(k,last_ps(k),:));
                                qp_table_main(k,last_ps(k),bi2de(last_cs_new(k,:))+1,pa(k))=median(qp_table_main(k,last_ps(k),bi2de(last_cs_new(k,:))+1,:));
                            end
                        end
                        break
                    end
                elseif length(find(last_ps==ps))~=noa
                    
                    stuck_counter=0;
%                 elseif training==0 & counter(i)>25
%                     temp_rew=0;
%                     break
                end
                
                %don't get stuck - if counter is bigger than 25 steps
                if training==0 & counter(i)>25
                    temp_rew=0;
                    break
                end


                % TERMINAL STATE CHECK                                  
                %check if we are in the terminal state and generate temp reward
                %which woul be used in table updating
                                                    %%%% make sure the ordering of
                                                    %%%% this module is
                                                    %%%% consistent
                                                    %%%% with its task, it should
                                                    %%%% update tables...
% %                 if ter>=1 
% %                                                   %calculating temp_reward
% %                                                   %this figure will be used to
% %                                                   %update q functions
% %                                           
% %                     temp_rew=1*best_rew^(ter-1)*gamma^counter(i);
% %                     rew_winner=[];
% %                     for ii=1:noa
% %                         if ps(ii)==n^2
% %                             rew_winner=[rew_winner,ii];
% %                         end  
% %                     end
% % 
% %                 end
                
                
                if ter==noa 
                                                  %calculating temp_reward
                                                  %this figure will be used to
                                                  %update q functions
                                          
                    rew_winner=[];
                    for ii=1:noa
                        if ps(ii)==goal_set
                            rew_winner=[rew_winner,ii];
                        end  
                    end
%                    temp_rew=best_rew^(length(rew_winner)-1);%*gamma^(counter(i));
                    temp_rew = best_rew;
                elseif ter >= 1 
                    rew_winner=[];
                    for ii=1:noa
                        if ps(ii)==goal_set
                            rew_winner=[rew_winner,ii];
                        end  
                    end
%                    temp_rew=best_rew^(length(rew_winner)-1);%*gamma^(counter(i));
                    temp_rew = 1;
                end
        
        
        
        
        
                %UPDATING TABLES:
                %updating communication table:
                %%%% when you have arrived to new episode, the last_ps
                %%%% would be the ending ps of the previous episode and
                %%%% gets unreal reward in this step this should be
                %%%% corrected

                %counting the number of steps in the current episode
                counter(i)=counter(i)+1;
            end

        %episode summerize
            if term_hist(i)==noa
                rew(i)=best_rew*gamma^(counter(i)-1);
            elseif term_hist(i)>=1
                rew(i)=1*gamma^(counter(i)-1);
            else
                rew(i)= 0; 
            end
% % % %             cumul_rew=cumul_rew+rew(i);
% % % %             saved_qc_8(i,:)=qc_table(1,n^2-1,:);
% % % %             saved_qc_6(i,:)=qc_table(1,n^2-n,:);
% % % %             saved_qp_8(i,:)=qp_table(1,n^2-1,1,:);
% % % %             saved_qp_6(i,:)=qp_table(1,n^2-n,1,:);
            if training ==1 & i>200
                probs_1 = sum(NE_table(1,:,:,:),4)/sum(sum(sum(NE_table(1,:,:,:),4)));
                exp_conditional_rew = sum(sum(max(qp_table_main(1,:,:,:),[],4).*probs_1)); %conditioned on the received communications and observations
                exp_rew(i) = sum(sum(exp_conditional_rew.*probs_1))*gamma^2;
            end
        end
        %save('qc3_table.mat','qc3_table')
        %save('qp3_table.mat','qp3_table')
        if training == 1
            save('qp_table_n4g16noa3_2.mat','qp_table_main')
        end

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

% 
% function cs_ind = cs2csind(cs,inf_bits)
% cs_ind = zeros(noa,1);
%     for i = 1:noa
%         temp = bi2de(cs(i,:,:));
%         cs_ind(i) = temp(1) + temp(2)*inf
%     end
% 
% end
function main_ps = ps_calc(ps,n,noa)

        main_ps = 0;
        for kk=1:noa
            if kk>1
                main_ps = main_ps+ (ps(kk)-1)*(n^2)^(kk-1);
            else
                main_ps = main_ps+ ps(kk);
            end
            
        end

end

function pa=pa_calc(main_pa,noa)
pa=zeros(noa,1);
%pa(1)=fix(main_pa/5)+ceil(rem(main_pa,5)/5);
%pa(2)=main_pa-(pa(1)-1)*5;

counter = 0;
    for kk = noa:-1:1
        main_pa = main_pa - counter;
        pa(kk) = ceil((main_pa)/(5^(kk-1)))  ;
        counter = (pa(kk)-1)*(5^(kk-1));
    end
end
        
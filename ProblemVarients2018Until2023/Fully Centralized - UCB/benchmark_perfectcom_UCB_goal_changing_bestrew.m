%% Numerical Simulator of Emergence of communication among rl agents under coordination environment
%Started: 05/03/2018
%Functions called : envir(ps,pa,n,noa) / envir_windy(ps,pa,n,noa,windy) , pdecide(ps,last_ps,cs,last_cs,scen,pa,temp_rew,rew_winner,tau), cdecide(ps,last_ps,ca,scen,temp_rew,rew_winner,tau)


clear
%close all
clc

%% Setup
%scen=;             
                    %communication scenario
n=4;                
                    %size of gridworld
noa=2;              
                    %number of agents
%ns=1000000;
ns=200000;
                    %number of simulations in each batch
bn=1;
                    %number of batchs

end_learn= 1; %0.850;

goal_set=16;

inf_bits=1;
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

s_space=1:1:n^2;
s_space(goal_set)=[];

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
        main_pa=randi(25);
        pa=pa_calc(main_pa);
                            %position action of each of agents (each row)
                            %done based on RL 
        ps_ind=randi(n^2-1,2,1);
        ps=zeros(noa,1);
        for i=1:noa
            ps(i) = s_space(ps_ind(i));
        end
        
        main_ps=(ps(1)-1)*n^2+ps(2);
        
                            %position state of each of agents (each row)
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
        qp_table=0.02*ones(n^4,25);
        N_table=0.001*ones(n^4,25);
        N_table_emerged=0.001*ones(n^4,25);
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
    
            %random initialization of position states and actions
            ps_ind=randi(n^2-1,2,1);
            ps=zeros(noa,1);
            for k=1:noa
                ps(k) = s_space(ps_ind(k));
            end
        
            main_ps=(ps(1)-1)*n^2+ps(2);
            main_pa=randi(25);
            pa=pa_calc(main_pa);
 
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
                [main_pa] = bench_policy_UCB(main_ps,0.005,qp_table,N_table,ns,i,ucb_counter,end_learn,best_rew);
                pa=pa_calc(main_pa);
                
                %Update UCB counter
                N_table(main_ps,main_pa)=N_table(main_ps,main_pa)+1;
                
                %Update UCB_emerged counter
                if i>end_learn*ns
                    N_table_emerged(main_ps,main_pa)=N_table_emerged(main_ps,main_pa)+1;
                end
                
                
                %canceling action if the agent is in the terminal state
                ps_check=ps_calc(main_ps,n);
                if ps_check(1)==goal_set || ps_check(2)==goal_set
                    main_pa=25;
                end
                
                                     
                %saving the previous position state before it is updated
                last_ps=main_ps;
                                       
                %OBTAIN NEW ENVIRONMENT STATES AND REWARD
                [ps,err,ter]=envir_gc(ps,pa,n,noa,goal_set); 
                main_ps=(ps(1)-1)*n^2+ps(2);

                
                
                if ter>=2 
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
                elseif ter==1
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
end
toc

        %% Visualization of last batch
        avg_len=100;
        avg_rew=zeros(ns-avg_len,1);
        for i=1:ns-avg_len
            avg_rew(i)=mean(rew(i:i+avg_len));
    
        end
        avg_counter=zeros(ns-avg_len,1);
        for i=1:ns-avg_len
            avg_counter(i)=mean(counter(i:i+avg_len));
    
        end    

        f=5;
        figure(f)
        hold on
        plot(mean(batch_rew,2))
        hold on
        plot(movmean(mean(batch_rew,2),ns*0.01))

        %% plot std
        std_mat=zeros(ns,1);
        for i=1:ns
            std_mat(i,1)=std(batch_rew(i,:));
        end
        
        
        f=6;
        figure(f)
        hold on
        y1=movmean(std_mat,ns*0.01);
        y2=movmean(mean(batch_rew,2),ns*0.01);
        figure
        x=1:length(y2);
        errorbar(x(1000:2000:end),y2(1000:2000:end),std_mat(1000:2000:end),'color',[1 230/256 230/256])
        hold on
        plot(x,y2,'color',[1 102/256 102/256])
        xlabel("Steps")
        ylabel("(Moving average applied on) rewards")
        title("Reward improvement through time - Fully centralized - UCB Policiy")
        grid on
        %set colors using https://www.rapidtables.com/web/color/RGB_Color.html
        %errorbar(x,y2,std_mat,'color',[1 230/256 230/256])
        
% % %         std_minus=ones(1,ns).* mean();
% % %         std_plus=ones(1,ns);
% % %         for i=1:bn
% % %             
% % % 
% % %         end

% % %         f=f+1;
% % %         figure(f)
% % %         hold on
% % %         plot(avg_counter)
% % %         xlabel("Episode number")
% % %         ylabel("Steps required to finish the episode")
% % %         grid minor
% % % 
% % %         f=f+1;
% % %         figure(f)
% % %         plot(saved_qc_8(:,1)-saved_qc_8(:,2))
% % %         xlabel("Steps")
% % %         ylabel("Communication-related decisions values")
% % %         title("Which communication action to take when in 8th grid")
% % % 
% % %         f=f+1;
% % %         figure(f)
% % %         plot(saved_qc_6(:,1))
% % %         hold on
% % %         plot(saved_qc_6(:,2))
% % %         xlabel("Steps")
% % %         ylabel("Communication-related decisions values")
% % %         title("Which communication action to take when in 6th grid")
% % %         legend("Send 1","Send 2")
% % % 
% % %         f=f+1;
% % %         figure(f)
% % %         for i=1:5
% % %             plot(saved_qp_8(:,i))
% % %             hold on
% % %         end
% % %         xlabel("Steps")
% % %         ylabel("Position-related decision values")
% % %         title("Which position action to take when in 8th grid")
% % %         legend("Go right", "Go left", "Go up", "Go down", "Stop")
% % % 
% % %         f=f+1;
% % %         figure(f)
% % %         for i=1:5
% % %             plot(saved_qp_6(:,i))
% % %             hold on
% % %         end
% % %         xlabel("Steps")
% % %         ylabel("Position-related decision values")
% % %         title("Which position action to take when in 6th grid")
% % %         legend("Go right", "Go left", "Go up", "Go down", "Stop")

% %% Inter Batch Visualization
% mean_rew=mean(batch_rew,2);
% mean_counter=mean(batch_counter,2);
% 
% %applying moving average on mean_rew
% avg_len=4000;
% mean_rew_mav=zeros(ns-avg_len,1);
% for i=1:ns-avg_len
%     mean_rew_mav(i)=mean(mean_rew(i:i+avg_len));
% 
% end
% 
% plot(mean_rew_mav)
% 
% figure
% plot(mean_counter)

%%%%%%%%%%%%%%%% State Aggregation %%%%%%%%%%%%%%%%%%%

%computing the V_o values and the probability of each o_1 under current
%policy
[V_o_1,N_o_1] = sum_q(qp_table,N_table_emerged,n);
V_o_1(:,2:end)=[];
%using Lloyd's algorithm to cluster observation values
[a,b]=lloyds(V_o_1(1:15,1),2^inf_bits);

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

%using Lloyd's algorithm for weighted V_o vector:
z_ind=find(V_o_1_weighted==0); %find the index of ending zero elements
if isempty(z_ind)
    V_o_1_weighted(z_ind(1)-1:end)=[]; %removing zero elements
end
[c,d]=lloyds(V_o_1_weighted(:,1),2^inf_bits);

%grouping states based on the group their value function is placed in:
ag_states=zeros(2^inf_bits,length(s_space));
cnt=ones(2^inf_bits,1);
c_aug=[c,max(V_o_1)]; %to facilitate next computations, we creat an augmented c in which the last elemnt is the max of V_o
for jj=1:length(s_space)
    for kk=1:2^inf_bits
        if V_o_1(s_space(jj))<=c_aug(kk)           
            ag_states(kk,cnt(kk))=s_space(jj);            
            cnt(kk)=cnt(kk)+1;
            break
        end
    end
end
%save('agreggated_states_g16_infbits3', 'ag_states') 


inf_bits=3;
syms = 5;
% State Aggregation by k-median clustering
[kmkm,C]=kmedoids(V_o_1_weighted,2^inf_bits,'Distance','euclidean');

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

%save('agreggated_states_g64_infbits2', 'ag_states') 
avg_rew=movmean(rew(1:110000),10000);
plot(1:100000/20:100000,avg_rew(1:length(avg_rew)/20:end))
% 
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

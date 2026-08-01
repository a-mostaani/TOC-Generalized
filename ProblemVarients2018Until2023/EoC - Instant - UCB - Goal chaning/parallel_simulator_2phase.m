%% Parallel simulator
clear
%close all
clc


%% Numerical Simulator of Emergence of communication among rl agents under coordination environment
%Started: 05/03/2018
%Functions called : envir(ps,pa,n,noa) / envir_windy(ps,pa,n,noa,windy) , pdecide(ps,last_ps,cs,last_cs,scen,pa,temp_rew,rew_winner,tau), cdecide(ps,last_ps,ca,scen,temp_rew,rew_winner,tau)


%use new functions of cpolicy cosumized cupdate customized and ...

scen=3;             
                    %communication scenario
n=4;                
                    %size of gridworld
noa=2;              
                    %number of agents
bits=3;
%ns=fix(55000*(2^(bits-7))); 
ns=25000;

                    %number of simulations in each epoch
                    %based on experience: 10k would be enough up to n=3
                    % 100K for n=4
                    % 200K for n=5
avg_len=ns*0.1;                    

en=4;
                    %number of epochs

err_trig=0.3;
best_rew=3;
worst_rew=1;
goal_set=1;

bsc_p_learn=0.05; 
bsc_p_exec=0.05;

end_learn_learn=0.85;
end_learn_exec=0.05;
%windy_envir        
                    %if you need a windy environment, you can choos windy_envir
                    %function instead of envir
update_tables=1;
%variables that can be modified in cdedcie():
%alpha=0.2; 
%sweep=0; %if sweep is one, we update the whole q_table at the time that the
         % function cdecide is called
         %if sweep is off, then only update the q_table for the current
         %state of the agents
gamma=0.9;


                    
tau_k=0.005;        
                    %the constant value based on which tau will be updated in each
                    %new episode
%term_hist=zeros(ns,1);

%h=waitbar(0,'Processing');

%% Zero initialization
epoch_rew=zeros(ns,en);
learning_rew=zeros(ns,en);
epoch_counter_learn=zeros(ns,en);
epoch_counter_exec=zeros(ns,en);

%disp(b)
%wind=zeros(1,2);
                    %if environment is windy, it can get 0 or 1 in x or y
                    %direction
%wind_loc           
                    %please note that if you want to change the setting
                    %related to the wind_loc you should do it in the
                    %initialization part of envir_windy


ca=zeros(noa,bits);      
                    %communication action of each of agents (each row)
                    %it can be specified based on communcation policy, there are
                    %two policies studied in this paper:
                    %1-sending the current position
                    %2-learning how to communicate using one bit of data

%policy             
                    %if you want to change the policy you should change it
                    %in cdecide or pdecide functions
cs=ones(noa,noa-1,bits);      
                    %communication state of each of agents (each row)
                    %This is equal to communication action of the other agents in
                    %the previous step (each column and its depth)
                    %since each agent might send more than one
                    %bits, the matrix contains depth, the matrix is
                    %as deep as the number bits sent by each agent
pa=randi(5,noa,1);      
                    %position action of each of agents (each row)
                    %done based on RL 
valid_loc=zeros(n*n-1,1); 
                    %This is a vector which includes only non-terminal
                    %locations of the grid so the initial locations can be
                    %selected from it.
ps=[0;0];                    
% % % %Here we select the location uniformly from the non-terminal locations of the greed                    
% % % %         flag_l=0;                    
% % % %         for i=1:n*n
% % % %             if i~=goal_set && flag_l==0
% % % %                 valid_loc(i)=i;
% % % %             elseif i~=goal_set && flag_l==1
% % % %                 valid_loc(i-1)=i;    
% % % %             else
% % % %                 flag_l=1;
% % % %             end
% % % %         end
% % % % 
% % % %         ps_1=randi(n*n-1,noa,1); 
% % % %                             %selecting the indices of the random initial locations
% % % %         ps=zeros(noa,1);
% % % %         for i=1:2                    
% % % %             ps(i)=valid_loc(ps_1(i));
% % % %         end
% % % %                     %position state of each of agents (each row)
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

%cumul_rew=0;

qc_table=0.02*ones(noa,n^2,2^bits);
qc=0.02*ones(noa,n^2,2^bits);
                    %qc3_table(n^2,:)=[2,2];
                    %the q table of communication actions for two
                    %agetns, qc_table(i,:,:) is qc table of ith agent
                    %based on the number of bits that can be sent
                    %by each agent, there are 2^b com actions that
                    %can be done by it.

switch scen
   case 1
       qp_table=0.02*ones(noa,n^2,2,5);
   case 2
       qp_table=0.02*ones(noa, n^2,n^2,5);
   case 3
       qp_table=0.02*ones(noa,n^2,2^bits,5);
       qp=0.02*ones(noa,n^2,2^bits,5);
end
                    %the q table of position actions for two agents
                    %qp_table(i,:,:,:) is qp table of ith agent


%load('qc3_table.mat','qc3_table')
%load('qp3_table.mat','qp3_table')

%saved_qc_8=zeros(ns,2^bits);
                        %saving the 8th row of qc table throughout time to
                        %to see how it evolves
%saved_qc_6=zeros(ns,2^bits);
%saved_qp_8=zeros(ns,5);
                        %saving the 8th row of qp table throughout time to
                        %to see how it evolves
%saved_qp_6=zeros(ns,5);

%% Main function








tic
parfor i=1:en
    %learning phase
    [learning_rew(:,i),qp,qc,epoch_counter_learn(:,i)]=EoC_corrected_UCB(scen,n,noa,ns,bits,best_rew,worst_rew,goal_set,gamma,tau_k,ca,cs,pa,ps,ter,rew,temp_rew,counter,qc_table,qp_table,bsc_p_learn,end_learn_learn,update_tables);
    
    %execution phase
    %[epoch_rew(:,i)]=EoC_corrected_UCB(scen,n,noa,ns,bits,best_rew,worst_rew,goal_set,gamma,tau_k,ca,cs,pa,ps,ter,rew,temp_rew,counter,qc,qp,bsc_p_exec,end_learn_exec,0);
    
end
toc









%% Visualization
%for i=1:ep
mean_rew_exec=mean(mean(epoch_rew(fix(ns*end_learn_exec+1):end,:)));
mean_rew_learn=mean(mean(learning_rew(fix(ns*end_learn_learn+1):end,:)));
%end
mean_rew_exec
mean_rew_learn
hold on
plot(mean(movmean(learning_rew,500),2))
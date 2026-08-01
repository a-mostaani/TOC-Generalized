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
ns=80000;              
                    %number of simulations in each batch
bn=1;
                    %number of batchs

end_learn=0.80;
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


                    
tau_k=0.005;        
                    %the constant value based on which tau will be updated in each
                    %new episode
gamma=.9;
%% Zero initialization
batch_rew=zeros(ns,bn);
batch_counter=zeros(ns,bn);
for b=1:bn 
        %disp(b)
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
        ps=randi(n^2-1,2,1);
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
            ps=randi(n^2-1,2,1);
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
                [main_pa] = bench_policy(main_ps,0.005,qp_table,ns,i,end_learn);
                pa=pa_calc(main_pa);
        
                %canceling action if the agent is in the terminal state

                if main_ps>=n^4
                    main_pa=25;
                end
                
                                     
                %saving the previous position state before it is updated
                last_ps=main_ps;
                                       
                %environment, position state update
                [ps,err,ter]=envir(ps,pa,n,noa); 
                main_ps=(ps(1)-1)*n+ps(2);

                
                
                if ter>=1 
                                                  %calculating temp_reward
                                                  %this figure will be used to
                                                  %update q functions
                          switch ter
                              case 1
                                  temp_rew=1*gamma^counter(i);
                              case 2
                                  temp_rew=6*gamma^counter(i);
                          end

                end
        
        
        
        
                %UPDATING TABLES:

                %counting the number of steps in the current episode
                counter(i)=counter(i)+1;

        
                %CHECKING WHILE LOOP CONDITION
                if ter >=1 
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

        %episode summerize
            rew(i)=temp_rew;
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

        %% Visualization of last batch
        avg_len=5000;
        avg_rew=zeros(ns-avg_len,1);
        for i=1:ns-avg_len
            avg_rew(i)=mean(rew(i:i+avg_len));
    
        end
        avg_counter=zeros(ns-avg_len,1);
        for i=1:ns-avg_len
            avg_counter(i)=mean(counter(i:i+avg_len));
    
        end    

        f=1;
        figure(f)
        hold on
        plot(avg_rew)
        xlabel("Steps")
        ylabel("(Moving average applied on) rewards")
        title("Reward improvement through time - scaled")
        grid minor

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
% 
%% calculate main 
function pa=pa_calc(main_pa)
pa=zeros(2,1);
pa(1)=fix(main_pa/5)+ceil(rem(main_pa,5)/5);
pa(2)=main_pa-(pa(1)-1)*5;
end

function ps=ps_calc(main_ps,n)
ps=zeros(2,1);
ps(1)=fix(main_ps/n)+ceil(rem(main_ps,n)/n);
ps(2)=main_ps-(ps(1)-1)*n;
end
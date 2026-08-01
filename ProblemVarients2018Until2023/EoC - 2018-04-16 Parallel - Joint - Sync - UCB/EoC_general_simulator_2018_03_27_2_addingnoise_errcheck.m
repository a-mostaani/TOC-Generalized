%% Numerical Simulator of Emergence of communication among rl agents under coordination environment
%Started: 05/03/2018
%Functions called : envir(ps,pa,n,noa) / envir_windy(ps,pa,n,noa,windy) , pdecide(ps,last_ps,cs,last_cs,scen,pa,temp_rew,rew_winner,tau), cdecide(ps,last_ps,ca,scen,temp_rew,rew_winner,tau)


%use new functions of cpolicy cosumized cupdate customized and ...

clear
%close all
clc

%% Setup

scen=3;             
                    %communication scenario
n=3;                
                    %size of gridworld
noa=2;              
                    %number of agents
ns=1000;              
                    %number of simulations in each epoch
                    %based on experience: 10k would be enough up to n=3
                    % 100K for n=4
                    % 200K for n=5
                    
avg_len=ns*0.1;        
                    %length of moving average
%ns=ns+avg_len;
                    
bits=2;
                    %number of bits that agents are allowd to communicate
                    %with
bsc_p=0.1;          %probability by which the binary symiteric channel 
                    %toggles the bit sent
en=1;
                    %number of epochs

best_rew=3;   

update_tables=0;

switch update_tables
    case 0
        end_learn_learn=0.05;
    case 1
        end_learn_learn=0.85;
    case 2
        end_learn_learn=input('After how many steps do you like the algorithm to become greedy?');
        end_learn_learn./ns;
end

%windy_envir        
                    %if you need a windy environment, you can choos windy_envir
                    %function instead of envir
update_upon_existing=0;
                    %if updates are done upon the existing q_functions or
                    %not the table based on which updating are taking place
                    %in %%update upon existing (search for it)

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
term_hist=zeros(ns,1);

h=waitbar(0,'Processing');
err_trig=0.35;
ch_err=zeros(ns,1); %number of communication errors occured at each episode
                    %of the last, or the only, epoch

%% Zero initialization
epoch_rew=zeros(ns,en);
epoch_counter=zeros(ns,en);
for b=1:en 
        %disp(b)
        wind=zeros(1,2);
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
                            
        cs_bc=ones(noa,noa-1,bits);
                            %communication state of each of agents before
                            %being mixed with noise, true cs
        cs=ones(noa,noa-1,bits);      
                            %communication state of each of agents
                            %this vector is polluted with noise
                            %(each row)
                            %This is equal to communication action of the other agents in
                            %the previous step (each column and its depth)
                            %since each agent might send more than one
                            %bits, the matrix contains depth, the matrix is
                            %as deep as the number bits sent by each agent
                            
        pa=randi(5,noa,1);      
                            %position action of each of agents (each row)
                            %done based on RL 
        ps=randi(n*n-1,noa,1);
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

        qc_table=0.02*ones(noa,n^2,2^bits);
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
       end
                            %the q table of position actions for two agents
                            %qp_table(i,:,:,:) is qp table of ith agent

       %update upon existing
       if update_tables==0     
%                 load('qc_table.mat','qc_table')
%                 load('qp_table.mat','qp_table')
                 load('qc_table_noisy.mat','qc_table')
                 load('qp_table_noisy.mat','qp_table')
                 load('qc_table_noisy_bugy.mat','qc_table')
                 load('qp_table_noisy_bugy.mat','qp_table')
       elseif update_tables==1 && update_upon_existing==1 %%update upon existing
                 load('qp_table_noisy_7bit_.2err_1.4.mat','qp_table')
                 load('qc_table_noisy_7bit_.2err_1.4.mat','qc_table')
       end
        saved_qc_8=zeros(ns,2^bits);
                                %saving the 8th row of qc table throughout time to
                                %to see how it evolves
        saved_qc_6=zeros(ns,2^bits);
        saved_qp_8=zeros(ns,5);
                                %saving the 8th row of qp table throughout time to
                                %to see how it evolves
        saved_qp_6=zeros(ns,5);

        %% Episode iteration
        for i=1:ns
            %Episode initialization
            %disp(i)
            waitbar(((b-1)/en)+i/(ns*b),h)
            temp_rew=0;
            rew_winner=[];
            counter(i)=1;

            % update tau, but above step 40'000 matlab will be unable to handle the
            % very big numbers so we don't go beyond...
            if i<=40000
                tau=1/(1+i*tau_k);
            else
                tau=1/(1+40000*tau_k); 
            end
    
            %random initialization of position states and actions
            ps=randi(n*n-1,noa,1);  
            pa=randi(5,noa,1);
    
            %random initialization of communication action ca, it depens on the scenario
            %selected
            switch scen
                case 1
                    ca=randi(2,noa,1);                             
                case 2
                    ca=ps;
                case 3
                    ca=randi(2,noa,1);
            end
    
        %episode run    
            while 3==3    
                        % why 3==3 ?
                        % while loop could not be conditioned on terminal state
                        % because after wev'e got to the terminal state still the 
                        % table updates should be done.
                        % Instead, at the end of each while loop it is checked if
                        % we've got to the terminal state or no
    
        
                
                
                %SELECT COMM. ACTION AND UPDATE COMM. STATE
                %select communication action
                [ca] = cpolicy_customized_nbits(ps,scen,tau,qc_table,i,ns,end_learn_learn);
                                                                
        
                %saving previouse communication state before it is updated
                last_cs=cs;
                %communication state update:
                %the previous ca of the other agent is the cs of the current
                %agent in case that no noise is added. Otherwise, noise
                %would change the signlas sent through the communication
                %channel.
                
                
                for j=1:noa
                    cs_temp=ca;
                    cs_temp(j,:)=[];
                    cs_bc(j,:,:)=cs_temp;
                end
                
                %Sending cs_beforechannel to channel 
                [cs,cherr]=bsc_ch(cs_bc,bsc_p);
                ch_err(i)=ch_err(i)+cherr;
                
                if update_tables==1
                    if counter(i)~=1   %%%%make sure if the if statement is necessary
                        %updating position table:
                        [qp_table] = pupdate_customized_nbits(ps,last_ps,cs,last_cs,scen,pa,temp_rew,rew_winner,tau,qp_table); 
                    end
                end
                %SELECT POSITION ACTION AND UPDATE POSITON STATE
                %select position action 
                [pa] = ppolicy_customized_nbits(ps,cs,scen,tau,qp_table,i,ns,end_learn_learn);
        
                %canceling action if the agent is in the terminal state
                for j=1:noa
                    if ps(j)==n^2
                        pa(j)=5;
                    end
                end
                                     
                %saving the previous position state before it is updated
                last_ps=ps;
                                       
                %environment, position state update
                [ps,err,ter]=envir(ps,pa,n,noa); 
                                                  %envir is a function which models the environment
                                                  %it tells if terminal state has
                                                  %been achieved, it tells if any
                                                  %error has occured
                                
                                          
                                          
                % TERMINAL STATE CHECK                                  
                %check if we are in the terminal state and generate temp reward
                %which woul be used in table updating
                                                    %%%% make sure the ordering of
                                                    %%%% this module is
                                                    %%%% consistent
                                                    %%%% with its task, it should
                                                    %%%% update tables...
                if ter>=1 
                                                  %calculating temp_reward
                                                  %this figure will be used to
                                                  %update q functions
                                          
                    temp_rew=1*best_rew^(ter-1)*gamma^counter(i);
                    rew_winner=[];
                    for ii=1:noa
                        if ps(ii)==n^2
                            rew_winner=[rew_winner,ii];
                        end  
                    end

                end
        
        
        
        
                %UPDATING TABLES:
                %updating communication table:
                %%%% when you have arrived to new episode, the last_ps
                %%%% would be the ending ps of the previous episode and
                %%%% gets unreal reward in this step this should be
                %%%% corrected
                if update_tables==1
                    if counter(i)==1
                    %no update

                    else
                            [qc_table] = cupdate_customized_nbits(ps,last_ps,ca,scen,temp_rew,rew_winner,qc_table);

                    end
                end
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
                    if update_tables==1
                        [qp_table] = pupdate_customized_nbits(ps,last_ps,cs,cs,scen,pa,temp_rew,rew_winner,tau,qp_table);
                    end
                    term_hist(i)=ter;
                    break
                end
            end

        %episode summerize
            if temp_rew>=1
                rew(i)=3*temp_rew/best_rew;
            else
                rew(i)=temp_rew;
            end
% % % %             cumul_rew=cumul_rew+rew(i);
% % % %             saved_qc_8(i,:)=qc_table(1,n^2-1,:);
% % % %             saved_qc_6(i,:)=qc_table(1,n^2-n,:);
% % % %             saved_qp_8(i,:)=qp_table(1,n^2-1,1,:);
% % % %             saved_qp_6(i,:)=qp_table(1,n^2-n,1,:);
        end
        %save('qc3_table.mat','qc3_table')
        %save('qp3_table.mat','qp3_table')

        
        
%epoch SAVING        
epoch_rew(:,b)=rew;
epoch_counter(:,b)=counter;

end

%Epoch reward
avg_epoch_rew=sum(epoch_rew,2)/en;

        %% Visualization of last epoch

        avg_rew=zeros(ns-avg_len,1);
        
        for i=1:ns-avg_len
            avg_rew(i)=mean(rew(i:i+avg_len));
    
        end
        
        avg_counter=zeros(ns-avg_len,1);
        for i=1:ns-avg_len
            avg_counter(i)=mean(counter(i:i+avg_len));
    
        end
        
        smooth_avg_epoch_rew=zeros(ns-avg_len,1);
        for i=1:ns-avg_len
            smooth_avg_epoch_rew(i)=mean(avg_epoch_rew(i:i+avg_len));
        end

        
%         f=1;
%         figure(f)
%         hold on
%         plot(avg_rew)
%         xlabel("Episodes")
%         ylabel("(Moving average applied on) rewards")
%         title("Reward improvement through time - scaled")
%         grid minor
% 
%         f=f+1;
%         figure(f)
%         hold on
%         plot(avg_counter)
%         xlabel("Episode number")
%         ylabel("Steps required to finish the episode")
%         grid minor
        
        %f=f+1;
        figure(1)
        hold on
        plot(smooth_avg_epoch_rew)
        xlabel("Episodes/10")
        title("Reward improvement through time - scaled")
        grid minor
        
        
        %plotting the the variance of reward at each certain episode number
        %over different epochs
        err_n=10; %number of points in which we evaluate the error
        neg=zeros(err_n,1);
        pos=zeros(err_n,1);
        for i=1:10
            xx=i*(ns-avg_len)/10;
            %neg=smooth_avg_epoch_rew(xx)-min(epoch_rew(xx,:));
            %pos=max(epoch_rew(xx,:))-smooth_avg_epoch_rew(xx);
            
            % finding positive error
            crit=0;
            up_bound=3;
            while crit==0
                ll=length(find(epoch_rew(xx,:)>=up_bound));
                crit=ll>=en*err_trig;
                up_bound=up_bound-.01;
            end
            up_bound=up_bound-smooth_avg_epoch_rew(xx);
            
            % finding negative error
            crit=0;
            low_bound=0;
            while crit==0
                ll=length(find(epoch_rew(xx,:)<=low_bound));
                crit=ll>=en*err_trig;
                low_bound=low_bound+.01;
            end
            low_bound=smooth_avg_epoch_rew(xx)-low_bound;
            %errorbar(xx,smooth_avg_epoch_rew(xx),low_bound,up_bound,'r')
        end
        
        classified_rew=zeros(10,2);
        for j=1:10
            found=find(ch_err==j-1);
            L=length(found);
            classified_rew(j,2)=L;
                for i=1:L
                    classified_rew(j,1)=classified_rew(j,1)+rew(found(i));
                end
        end
        norm_classified_rew=classified_rew(:,1)./classified_rew(:,2);
        figure(2)
        hold on
        plot(0:9,norm_classified_rew)
        xlabel('Total number of communication errors occurred in an episode')
        ylabel('Average reward taken')
        grid minor
% % 
% %         f=f+1;
% %         figure(f)
% %         plot(saved_qc_8(:,1)-saved_qc_8(:,2))
% %         xlabel("Steps")
% %         ylabel("Communication-related decisions values")
% %         title("Which communication action to take when in 8th grid")
% % 
% %         f=f+1;
% %         figure(f)
% %         plot(saved_qc_6(:,1))
% %         hold on
% %         plot(saved_qc_6(:,2))
% %         xlabel("Steps")
% %         ylabel("Communication-related decisions values")
% %         title("Which communication action to take when in 6th grid")
% %         legend("Send 1","Send 2")
% % 
% %         f=f+1;
% %         figure(f)
% %         for i=1:5
% %             plot(saved_qp_8(:,i))
% %             hold on
% %         end
% %         xlabel("Steps")
% %         ylabel("Position-related decision values")
% %         title("Which position action to take when in 8th grid")
% %         legend("Go right", "Go left", "Go up", "Go down", "Stop")
% % 
% %         f=f+1;
% %         figure(f)
% %         for i=1:5
% %             plot(saved_qp_6(:,i))
% %             hold on
% %         end
% %         xlabel("Steps")
% %         ylabel("Position-related decision values")
% %         title("Which position action to take when in 6th grid")
% %         legend("Go right", "Go left", "Go up", "Go down", "Stop")

% %% Inter epoch Visualization
% mean_rew=mean(epoch_rew,2);
% mean_counter=mean(epoch_counter,2);
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

if update_tables==1 && bsc_p==0
    save('qc_table.mat','qc_table')
    save('qp_table.mat','qp_table')
elseif update_tables==1 && bsc_p==0.1
    save('qc_table_noisy.mat','qc_table')
    save('qp_table_noisy.mat','qp_table')
elseif update_tables==1 && bsc_p==0.2
    save('qc_table_noisy2.mat','qc_table')
    save('qp_table_noisy2.mat','qp_table')
end

mean_rew_exec=mean(mean(epoch_rew(fix(ns*end_learn_learn+1):end,:)));
mean_rew_exec
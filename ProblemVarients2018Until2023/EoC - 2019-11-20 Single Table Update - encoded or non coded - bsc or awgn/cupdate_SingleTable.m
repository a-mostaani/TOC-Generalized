function [qc_table] = cupdate_SingleTable(ps,last_ps,ca,scen,temp_rew,rew_winner,qc_table)
%The function takes the position of the agent and gives the best
%communication action based on the scenarion selected.


%      This function is made out of two parts:
%      1-A Q function approximator (an updating lookup table)
%      2-A predefined policy
%Based on the policy and the Q value of each action-state, the best
%communcation action is selected and returened



%value initialization
%load('qc3_table.mat','qc3_table'); %the table is intialized in general simulator qc3_table=zeros(n^2,2)
alpha=0.05; 
sweep=0; %if sweep is one, we update the whole q_table at the time that the
         % function cdecide is called
         %if sweep is off, then only update the q_table for the current
         %state of the agents
gamma=0.9;
%tau is set from the mother function


%variable initialization
noa=length(ps);

tobe_sent=zeros(noa,1); %in order to reduce the number of calculation of this vector, or matrix in case of noa>2
for i=1:noa
    tobe_sent(i,1)=bi2de(ca(i,:))+1;
end

%updating q_table: (for scenario 1 and 2 no q_talbe is required and the
%action is selected directly) 
switch scen
    case 1
        ca=ones(noa,1);
        return
    case 2
        return
        
    case 3
        %sspace=1:1:n^2; %being in any of positions
        %aspace=[1,noa]; %sending 1 or 2
        %qc3_table=zeros(n^2,noa); 
        switch sweep
            case 0 %update the q_table only for the current state-action
% %                 similar to what has been mentioned in the paper
% %                 for i=1:noa
% %                     if ismember(i,rew_winner)
% %                         qc_table(i,last_ps(i),ca(i))=(1-alpha)*qc_table(i,last_ps(i),ca(i))+alpha*(temp_rew+gamma*best_qc3(ps,i,qc_table(i,:,:)));
% %                     else
% %                         qc_table(i,last_ps(i),ca(i))=(1-alpha)*qc_table(i,last_ps(i),ca(i))+alpha*(0+gamma*best_qc3(ps,i,qc_table(i,:,:)));
% %                     end
% %                 end
                    if length(rew_winner)==2
                        for i=1:noa
                            qc_table(last_ps(i),tobe_sent(i,1))=(1-alpha)*qc_table(last_ps(i),tobe_sent(i,1))+alpha*(temp_rew); %+gamma*best_qc3(ps,i,qc_table(i,:,:))
                        end
                    elseif length(rew_winner)==1
                        for i=1:noa
                            %qc_table(i,last_ps(i),tobe_sent(i,1))=(1-alpha)*qc_table(i,last_ps(i),tobe_sent(i,1))+alpha*(-1); %+gamma*best_qc3(ps,i,qc_table(i,:,:))
                            qc_table(last_ps(i),tobe_sent(i,1))=(1-alpha)*qc_table(last_ps(i),tobe_sent(i,1))+alpha*(temp_rew);
                        end
                    else
                        for i=1:noa
                            qc_table(last_ps(i),tobe_sent(i,1))=(1-alpha)*qc_table(last_ps(i),tobe_sent(i,1))+alpha*(0+gamma*best_qc3(ps,i,qc_table(:,:)));
                        end
                    end
                
            case 1 %update as much as you can
                for i=1:noa
                    
                end
                
        
        end
%        save('qc3_table.mat','qc3_table');
        
end
end

%Local function: find best action for a certain state
function q = best_qc3(ps,i,qci_table)
    q=max(qci_table(ps(i),:));
end




function [q_table] = jupdate_customized_nbits_2(ps,last_ps,cs,last_cs,scen,pa,ca,temp_rew,rew_winner,tau,q_table)
%General help


%      Detailed help



%Initialize
%load('qp3_table.mat','qp3_table'); %the table is intialized in general simulator qc3_table=zeros(n^2,2)
alpha=0.020; 
sweep=0; %if sweep is one, we update the whole q_table at the time that the
         % function cdecide is called
         %if sweep is off, then only update the q_table for the current
         %state of the agents
gamma=0.9;
%tau is set from the mother function

%variable initialization
noa=length(ps);
%[~,~,d1,~,~]=size(q_table);
%bits=log2(d1);
data_received=zeros(noa,1);
for i=1:noa
    data_received(i,1)=bi2de(transpose(last_cs(:,i)))+1;
end

%updating q_table:
switch scen
    case 1
        [qp_table] = pupdate_customized_nbit(ps,last_ps,cs,last_cs,3,pa,temp_rew,rew_winner,tau,qp_table);   %%%%position action should be taken in the ppolicy
    case 2
        [qp_table] = pupdate_customized_nbit(ps,last_ps,cs,last_cs,3,pa,temp_rew,rew_winner,tau,qp_table);
        
    case 3

        switch sweep
            case 0 %update the q_table only for the current state-action (SARSA)
               if temp_rew==0
                    for i=1:noa
%                         if ismember(i,rew_winner)
%                              qp_table(i,last_ps(i),last_cs(i),pa(i))=(1-alpha)*qp_table(i,last_ps(i),last_cs(i),pa(i))+alpha*(temp_rew+gamma*best_qp(ps,i,cs,qp_table(i,:,:,:)));
%                         else
                             %if you have more than two agents, this
                             %statement should be changed: #noa
                             
                                                  %  converting binary cs
                                                  %  to an address in the
                                                  %  table
                                                  %          |            
                                                  %          |
                                                  %          |
                                                  %          |
                                                  %          |
                              q_table(i,last_ps(i),data_received(i,1),pa(i),bi2de(transpose(ca(:,i)))+1)=(1-alpha)*q_table(i,last_ps(i),data_received(i,1),pa(i),bi2de(transpose(ca(:,i)))+1)+alpha*(0+gamma*max(q_table(i,ps(i),bi2de(transpose(cs(:,i)))+1,:,:),[],'all'));
%                         end
                    end
               elseif temp_rew~=0 && length(rew_winner)==1
                    for i=1:noa
                         if ismember(i,rew_winner)
                              q_table(i,last_ps(i),data_received(i,1),pa(i),bi2de(transpose(ca(:,i)))+1)=(1-alpha)*q_table(i,last_ps(i),data_received(i,1),pa(i),bi2de(transpose(ca(:,i)))+1)+alpha*(temp_rew);
                              %qp_table(i,last_ps(i),data_received(i,1),pa(i))=(1-alpha)*qp_table(i,last_ps(i),data_received(i,1),pa(i))+alpha*(-0.5);
                         else
                              %qp_table(i,last_ps(i),data_received(i,1),pa(i))=(1-alpha)*qp_table(i,last_ps(i),data_received(i,1),pa(i))+alpha*(0+gamma*max(qp_table(i,ps(i),bi2de(transpose(squeeze(cs(i,1,:))))+1,:)));
                              q_table(i,last_ps(i),data_received(i,1),pa(i),bi2de(transpose(ca(:,i)))+1)=(1-alpha)*q_table(i,last_ps(i),data_received(i,1),pa(i),bi2de(transpose(ca(:,i)))+1)+alpha*(temp_rew);

                         end
                    end
               else
                    for i=1:noa
                              q_table(i,last_ps(i),data_received(i,1),pa(i),bi2de(transpose(ca(:,i)))+1)=(1-alpha)*q_table(i,last_ps(i),data_received(i,1),pa(i),bi2de(transpose(ca(:,i)))+1)+alpha*(temp_rew);
                    end
                   
               end
                
            case 1 %update as much as you can
                for i=1:noa
                    
                end
                
        
        end
        %save('qp3_table.mat','qp3_table');
        
end

end






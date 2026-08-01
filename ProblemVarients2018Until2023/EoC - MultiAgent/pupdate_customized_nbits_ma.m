function [qp_table] = pupdate_customized_nbits_ma(ps,last_ps,cs,last_cs,scen,pa,temp_rew,rew_winner,tau,qp_table)
%General help


%      Detailed help



%Initialize
%load('qp3_table.mat','qp3_table'); %the table is intialized in general simulator qc3_table=zeros(n^2,2)
alpha=0.07; 
sweep=0; %if sweep is one, we update the whole q_table at the time that the
         % function cdecide is called
         %if sweep is off, then only update the q_table for the current
         %state of the agents
gamma=0.9;
%tau is set from the mother function

%variable initialization
noa=length(ps);
[~,~,d1,~]=size(qp_table);
rec_bits_per_agent=log2(d1);
[~,~,bits_sent_per_agent] = size(last_cs); %does not work for heterogeneous bit rates
last_data_received=zeros(noa,1);
data_received=zeros(noa,1);

%creating a new received communication matrix where an agents received
%message is all the messages received from the other agents stacked one
%after each other

last_cs_new=zeros(noa,rec_bits_per_agent);
% for i=1:noa %for multi agent you should extend this expression (works only for 3 agents)
%     last_cs_new(i,:)=[squeeze(last_cs(i,1,:)); squeeze(last_cs(i,2,:))];
% end
bit_counter = 0;
 for i=1:noa 
     bit_counter = 0;
     for j = 1:noa-1
         last_cs_new(i,bit_counter + 1 : bit_counter + bits_sent_per_agent)=squeeze(last_cs(i,j,:));
         bit_counter = bit_counter + bits_sent_per_agent;  %does not work for heterogeneous bit rates
     end
 end

cs_new=zeros(noa,rec_bits_per_agent);
 for i=1:noa 
     bit_counter = 0;
     for j = 1:noa-1
         cs_new(i,bit_counter + 1 : bit_counter + bits_sent_per_agent)=squeeze(cs(i,j,:));
         bit_counter = bit_counter + bits_sent_per_agent;  %does not work for heterogeneous bit rates
     end
 end


if noa > 2
    for i=1:noa
        last_data_received(i,1)=bi2de(last_cs_new(i,:))+1;
        %data_received(i,1)=bi2de(last_cs(i,1,:))+1;
    end    
elseif noa==2
    for i=1:noa
        last_data_received(i,1)=bi2de(transpose(squeeze(last_cs(i,1,:))))+1;
        %data_received(i,1)=bi2de(last_cs(i,1,:))+1;
    end    
end

if noa > 2
    for i=1:noa
        data_received(i,1)=bi2de(cs_new(i,:))+1;
        %data_received(i,1)=bi2de(last_cs(i,1,:))+1;
    end    
elseif noa==2
    for i=1:noa
        data_received(i,1)=bi2de(transpose(squeeze(last_cs(i,1,:))))+1;
        %data_received(i,1)=bi2de(last_cs(i,1,:))+1;
    end    
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
                              qp_table(i,last_ps(i),last_data_received(i,1),pa(i))=(1-alpha)*qp_table(i,last_ps(i),last_data_received(i,1),pa(i))+alpha*(0+gamma*max(qp_table(i,ps(i),bi2de(cs_new(i,:))+1,:)));
%                         end
                    end
               elseif temp_rew~=0 && length(rew_winner)==noa
                    for i=1:noa
                         if ismember(i,rew_winner)
                              qp_table(i,last_ps(i),last_data_received(i,1),pa(i))=(1-alpha)*qp_table(i,last_ps(i),last_data_received(i,1),pa(i))+alpha*(temp_rew);
                              %qp_table(i,last_ps(i),data_received(i,1),pa(i))=(1-alpha)*qp_table(i,last_ps(i),data_received(i,1),pa(i))+alpha*(-0.5);
                         else
                              %qp_table(i,last_ps(i),data_received(i,1),pa(i))=(1-alpha)*qp_table(i,last_ps(i),data_received(i,1),pa(i))+alpha*(0+gamma*max(qp_table(i,ps(i),bi2de(transpose(squeeze(cs(i,1,:))))+1,:)));
                              qp_table(i,last_ps(i),last_data_received(i,1),pa(i))=(1-alpha)*qp_table(i,last_ps(i),last_data_received(i,1),pa(i))+alpha*(temp_rew);

                         end
                    end
               else
                    for i=1:noa
                              qp_table(i,last_ps(i),last_data_received(i,1),pa(i))=(1-alpha)*qp_table(i,last_ps(i),last_data_received(i,1),pa(i))+alpha*(temp_rew);
                    end
                   
               end
                
            case 1 %update as much as you can
                for i=1:noa
                    
                end
                
        
        end
        %save('qp3_table.mat','qp3_table');
        
end

end






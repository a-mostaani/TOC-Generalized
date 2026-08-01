function [qp_table] = pbench_update(ps,last_ps,pa,temp_rew,qp_table)
%General help


%      qp_table=0.5*ones(n^2,25);



%Initialize
%load('qp3_table.mat','qp3_table'); %the table is intialized in general simulator qc3_table=zeros(n^2,2)
alpha=0.07; 

gamma=0.9;
%tau is set from the mother function


%updating q_table:

if temp_rew==0
           
    qp_table(last_ps,pa)=(1-alpha)*qp_table(last_ps,pa)+alpha*(gamma*max(qp_table(ps,:)));
else
    qp_table(last_ps,pa)=(1-alpha)*qp_table(last_ps,pa)+alpha*(temp_rew);

end
               
%save('qp3_table.mat','qp3_table');
        


end








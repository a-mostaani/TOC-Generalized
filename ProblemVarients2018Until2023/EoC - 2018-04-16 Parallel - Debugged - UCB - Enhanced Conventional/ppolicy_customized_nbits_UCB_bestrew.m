function [pa] = ppolicy_customized_nbits_UCB_bestrew(ps,cs,scen,tau,qp_table,NE_table,ucb_counter,ns,end_learn,best_rew)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

%INITIALIZATION
policy="ucb";
ucb_const=1.25*best_rew/3;

noa=length(ps);
pa=zeros(noa,1);
sum_q=0;
[noa,sth,bits2,acts]=size(qp_table);
bits=log2(bits2);
p_paction=zeros(2,1); %probability of each communication action
%qp3_table=0.5*ones(noa,n^2,2,5);
                    %the q table of position actions for sceneario 3
                    
%calculating epsilon:
if fix(end_learn*ns/100)~=0
    coef=.4/(fix(end_learn*ns/100));
else
    coef=0.4;  %equivalently epsilon has become zero
end
%epsilon=max([.4-coef*fix(counter/100),0.01]);
epsilon=0.02;

%Action Selection (Policy)

switch scen
    case 1
        
    case 2       
        %ca(i)=ps(i);
        [pa] = ppolicy_customized(ps,cs,3,tau,qp_table,counter,ns);
    case 3
        switch policy
            case "ucb"
                ucb_func=zeros(5,2);
                for i=1:noa
                    ucb_func(:,i)=ucb_const*sqrt(log(ucb_counter+1)./NE_table(i,ps(i),bi2de(zero_fix(cs(i,1,:),bits))+1,:));
                    [~,pa(i)]=max(squeeze(qp_table(i,ps(i),bi2de(zero_fix(cs(i,1,:),bits))+1,:))+ucb_func(:,i));
                end
            case "ep_greedy"
                for i=1:noa
                    rr=rand;
                    if rr<=epsilon
                        pa(i)=randi(5);
                    else
                        [val,pa(i)]=max(qp_table(i,ps(i),bi2de(zero_fix(cs(i,1,:),bits))+1,:));
                    end
                end
            case "q_prob"
                %decision making for each agent
                for i=1:noa
                    %change the scale of q (in the desired part):
                    resc_q=exp(qp_table(i,ps(i),bi2de(cs(i,1,:))+1,:)/tau);
                    %evaluating sum of exp(q/tau) values for any possible action
                    sum_q=sum(resc_q);
                    %changing the scale of sum q to 100
                    sum_q=100/sum_q;
                    %evaulating proportion of each q value compared with
                    %sum
                    p_paction=sum_q*resc_q;
                    if length(find(resc_q==inf))>=1
                        switch length(find(resc_q==inf))
                            case 1
                                ind=find(resc_q==inf);
                                p_paction(ind)=100;
                            case 2 || 3 || 4 || 5
                                error('More than one infinity value in action probability vect')
                        end
                        
                    end
                    rr=rand;
                    if rr<p_paction(1)/100
                        pa(i)=1;
                    elseif rr<sum(p_paction(1:2))/100
                        pa(i)=2;
                    elseif rr<sum(p_paction(1:3))/100
                        pa(i)=3;
                    elseif rr<sum(p_paction(1:4))/100
                        pa(i)=4;
                    else
                        pa(i)=5;
                        
                    end
                end
                    
                
        end
end

end

function out = zero_fix(ca_temp,bits)
    l=length(ca_temp);
    out=zeros(1,bits);
    for i=1:l
        out(i)=ca_temp(i);
    end
end





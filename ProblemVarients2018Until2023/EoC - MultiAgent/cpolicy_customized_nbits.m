function [ca] = cpolicy_customized_nbits(ps,scen,tau,qc_table,counter,ns,end_learn)

%The function calculates the ca based on qc_table OF EACH OF AGENTS
%   as many as number of agents, there should be qc tables given to the
%   function as input


policy="ep_greedy";
sum_q=0;
p_caction=zeros(2,1); %probability of each communication action
noa= length(ps);

%calculating epsilon for epsilon greedy policy
%calculating epsilon:
if fix(end_learn*ns/100)~=0
    coef=.4/(fix(end_learn*ns/100));
else
    coef=0.4;  %equivalently epsilon has become zero
end


%epsilon=max([.4-coef*fix(counter/100),0.01]);
epsilon=0.015;

[w,l,d]=size(qc_table);
bits=log2(d);
ca=zeros(noa,bits);
%Action Selection (Policy)

switch scen
    case 1
        
    case 2       
        ca=ps;
        return
    case 3
        switch policy
            case "greedy"
                for i=1:noa
                     ca(i)=best_qc3(ps,i,qc_table(i,:,:));
                end
            case "ep_greedy"
                for i=1:noa
                    rr=rand;
                    if rr<=epsilon %random selection
                        ca_temp=de2bi(randi(2^bits)-1);
                        ca(i,:)=zero_fix(ca_temp,bits);
                    else %greedy selection
                        [val,ind]=max(qc_table(i,ps(i),:));
                        ca(i,:)=zero_fix(de2bi(ind-1),bits);
                    end
                end
            case "q_prob"
                %decision making for each agent
                %In this part the nbit com. has not been implemented yet
                for i=1:noa
                    %change the scale of q (in the desired part):
                    resc_q=exp(qc_table(i,ps(i),:)/tau);
                    %evaluating sum of exp(q/tau) values for any possible action
                    sum_q=sum(resc_q);
                    %changing the scale of sum q to 100
                    sum_q=100/sum_q;
                    %evaulating proportion of each q value compared with
                    %sum
                    p_caction=sum_q*resc_q;
                    rr=rand;
                    if rr <= p_caction(1)/100
                        ca(i)=1;
                    else
                        ca(i)=2;
                    end
                end
                    
                
        end
end
end



%Local function: find best action for a certain state
function q = best_qc3(ps,i,qci_table)
    q=max(qci_table(ps(i),:));
end

function out = zero_fix(ca_temp,bits)
    l=length(ca_temp);
    out=zeros(1,bits);
    for i=1:l
        out(i)=ca_temp(i);
    end
end


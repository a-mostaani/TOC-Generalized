function [pa,ca] = jpolicy_approximated_UCB(ps,cs,scen,tau,q_table,N_table,ucb_counter,~,~)
%Joint policy: Selection of domain level and communication actions based on
%joint policy

%   Q-table is fetched
%   N-table is fetched
%   UCB function which is the Q-value of each state-action is calculated 
%       based on the UCB policy
%   The action which maximizes the UCB policy is returened

%INITIALIZATION
policy="ucb";
ucb_const=5;

noa=length(ps);
pa=zeros(noa,1);
sum_q=0;
[noa,~,bits2,~,~]=size(q_table);
bits=log2(bits2);
ica=zeros(noa,1); %numerical equivalent of the selected ca (communication action)
ca=zeros(bits,noa);


%p_paction=zeros(2,1); %probability of each communication action
%qp3_table=0.5*ones(noa,n^2,2,5);
                    %the q table of position actions for sceneario 3
                    
%calculating epsilon:
% if fix(end_learn*ns/100)~=0
%     coef=.4/(fix(end_learn*ns/100));
% else
%     coef=0.4;  %equivalently epsilon has become zero
% end
%epsilon=max([.4-coef*fix(counter/100),0.01]);
epsilon=0.02;

%Action Selection (Policy)

switch scen
    case 1
        
    case 2       
        %ca(i)=ps(i);
        %[pa] = ppolicy_customized(ps,cs,3,tau,qp_table,counter,ns);
    case 3
        switch policy
            case "ucb"
                ucb_func=zeros(5,2^bits,noa);
                for i=1:noa
                    ucb_func(:,:,i)=ucb_const*sqrt(log(ucb_counter+1)./N_table(i,ps(i),bi2de(transpose(cs(:,i)))+1,:,:));
                    
                    
                    %[r,c,v]=find(X==max(X,[],'all')) to find maximum value
                    %of X and its r(ow) and c(olumn) indices
                    %X: squeeze(q_table(i,ps(i),bi2de(cs(:,i))+1,:,:))+ucb_func(:,:,i)
                    X=squeeze(q_table(i,ps(i),bi2de(transpose(cs(:,i)))+1,:,:))+ucb_func(:,:,i);
                    [rows,cols,~]=find(X==max(X,[],'all'));
                    leng=length(rows);
                    ran_ind=randi(leng);
                    pa(i,1)=rows(randi(ran_ind));
                    ica(i,1)=cols(randi(ran_ind));
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

for i=1:2
    ca(:,i)=de2bi(ica(i)-1,bits);
    
end


end

%This function input: [0,1,0] --------> [0,1,0,0] ,where B=4
function out = zero_fix(ca_temp,bits)
    l=length(ca_temp);
    out=zeros(1,bits);
    for i=1:l
        out(i)=ca_temp(i);
    end
end





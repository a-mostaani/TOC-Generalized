function [pa] = bench_policy_UCB(ps,tau,qp_table,N_table,ns,episode,ucb_counter,end_learn,best_rew)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

%INITIALIZATION
[n4,~]=size(N_table);
n=sqrt(sqrt(n4));
ucb_const=0.75*best_rew/3;
noa=length(ps);
pa=zeros(noa,1);
sum_q=0;
p_paction=zeros(0); %probability of each communication action
%qp3_table=0.5*ones(n^2,2,5);
                    %the q table of position actions for sceneario 3
                    
% % if episode>=10000
% %     ucb_const=ucb_const-2*(episode-10000)/20000;
% %     ucb_const=max(0,ucb_const);
% % end
policy='ucb';

update= episode<ns*end_learn;

% %calculating epsilon for epsilon greedy policy
% %calculating epsilon:
% if fix(end_learn*ns/100)~=0
%     coef=.4/(fix(end_learn*ns/100));
% else
%     coef=0.4;  %equivalently epsilon has become zero
% end
% 
% 
% epsilon=max([.4-coef*fix(episode/100),0.001]);


%epsilon=.4-coef*fix(counter/100);


%calculating epsilon:
% % if fix(end_learn*ns/100)~=0
% %     coef=.6/(fix(end_learn*ns/100));
% % else
% %     coef=0.6;  %equivalently epsilon has become zero
% % end
%epsilon=max([.6-coef*fix(counter/100),0.01]);
% epsilon=0.1;

init_epsilon = 1;
epsilon = init_epsilon - episode/(end_learn*ns);


%Action Selection (Policy)


switch policy
    case 'stochastic'
        %change the scale of q (in the desired part):
        resc_q=exp(qp_table(ps(1),ps(2),:,:)/tau);
        %evaluating sum of exp(q/tau) values for any possible action
        sum_q=sum(sum(resc_q));
        %changing the scale of sum q to 100
        sum_q=100/sum_q;
        %evaulating proportion of each q value compared with
        %sum
        p_paction=sum_q*resc_q;


        rr=rand;
        summand=0;
        for j=1:5 
            for i=1:5 %searching whole action space
                summand=summand+p_paction(1,1,i,j);    
                if rr*100<=summand
                    pa=[i,j];
                    return
                end


            end
        end
    case 'stochastic-ucb'
        if update==1
            %change the scale of q (in the desired part):
            ucb_func=ucb_const*sqrt(log(ucb_counter)./N_table);
            resc_q=exp((qp_table(ps,:,:)+ucb_func(ps,:))/tau);
            %evaluating sum of exp(q/tau) values for any possible action
            sum_q=sum(sum(resc_q));
            %changing the scale of sum q to 100
            sum_q=100/sum_q;
            %evaulating proportion of each q value compared with
            %sum
            p_paction=sum_q*resc_q;


            rr=rand;
            summand=0;
            for j=1:25 
                summand=summand+p_paction(j);    
                if rr*100<=summand
                    pa=j;
                    return
                end
            end
        else
            [~,pa]=max(qp_table(ps,:));
        end
        
        
    case 'stochastic-epsilon'
        %According to:
        %Value-Difference based Exploration: Adaptive Control between 
        %epsilon-Greedy and Softmax
        % http://tokic.com/www/tokicm/publikationen/papers/KI2011.pdf
        if update==1
            ran = rand;
            if ran<epsilon
                %change the scale of q (in the desired part):
                resc_q=exp((qp_table(ps,:,:))/tau);
                %evaluating sum of exp(q/tau) values for any possible action
                sum_q=sum(sum(resc_q));
                %changing the scale of sum q to 100
                sum_q=100/sum_q;
                %evaulating proportion of each q value compared with
                %sum
                p_paction=sum_q*resc_q;


                rr=rand;
                summand=0;
                for j=1:25 
                    summand=summand+p_paction(j);    
                    if rr*100<=summand
                        pa=j;
                        return
                    end
                end
            else %if the rand number is less than epsilon we select actions 
                % greedily
                [~,pa]=max(qp_table(ps,:));
            end
        else       %if training phase if finished we select actions greedily
            [~,pa]=max(qp_table(ps,:));
        end    
    case 'ep-greedy'
        ran=rand;
        if ran<epsilon
            pa=randi(25);
        else
            [~,pa]=max(qp_table(ps,:));
        end
        
        
    case 'ucb'        
        ucb_func=zeros((n^2-1)^2,25);
        if update==1
            ucb_func=ucb_const*sqrt(log(ucb_counter)./N_table);
            [~,pa]=max(qp_table(ps,:)+ucb_func(ps,:));
        else
            [~,pa]=max(qp_table(ps,:));
        end
end

end




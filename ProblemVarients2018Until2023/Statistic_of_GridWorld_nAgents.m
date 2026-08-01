
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Understanding the statistic of the gridworld problem%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
close all
clc


%% Initialization of control parameters
N = 3; % the width and the length of the gridworld
G = 8; % goal point location (first grid is numbered zero)
NI = 1.5*10^6; % Number of Iterations
gamma = 0.9; % the discount factor
max_rew = 10; % the reward that can be attained if the task is accomplished
min_rew = 1; % the reward that is obtained if no communication is in place

%% Non controlable paramters
noa = 4; % number of agents

%% Zero Initialization
l_num = zeros(NI,noa); %Agent location numbers at all iterations
l = zeros(NI, noa, 2); %Agent coordinates at all iterations
d = zeros(NI, 1);
d_min = zeros(NI, 1);
counter = 0;



%% Experimenting
%We compute coordinates of the goal point:
G_coord = num_to_coord(G, N);

%Generate a set of location numbers that does not include the goal point
%location
legit_locs = 0:1:N^2-1;
legit_locs(G+1) = [];




%We generate the random coordinates of the K agents in the gridworld for NS number of times:
for i = 1:NI
    %generate the coordinates of K=2 agents at iteration N (not being placed on the goal point)
    for j=1:noa
        l_num(i,j) = randsample(legit_locs,1);
    end
    
    %convert location numbers to location coordinates
    for j=1:noa
        l(i,j,:) = num_to_coord(l_num(i,j), N);
    end

end

% for i = 1:NI
%     for j=1:K
%         %generate the coordinates of K=2 agents at iteration N (not being placed on the goal point)
%         l_num(i,K) = randsample(legit_locs,1);
%     end
%     
%     for j=1:K
%         %convert location numbers to location coordinates
%         l(i,K,:) = num_to_coord(l_num(i,K), N);
%     end
% 
% end


%We compute the maximum distance of the agents from the goal point at each
%single iteration
for i = 1:NI
    [d(i),d_min(i)] = compute_distance(G_coord, l(i,:,:));
end



%% Summerizing the experiment
% obtaining the probability distribution of episode length
h = histogram(d);
[count, edges] = histcounts(d);
prob = count/NI;

% computing the expeted reward
range = length(prob);
discount_vect = ones(1,range);
for i = 1:range
    discount_vect(i) = gamma^(i);
end
rew_vect = discount_vect * max_rew;
expected_rew = rew_vect * transpose(prob);
variance = ((rew_vect - expected_rew).^2)*transpose(prob);

% %plotting performance
% x = 1 : 200000/20 : 200000;
% ll = length(x);
% figure
% plot(x , expected_rew*ones(1, ll))
% 
% 
% %calculating joint probabilities for w^t = 22:
% %prob dict:
% prob_dict = [1, 4; 2, 7; 3, 8; 4, 8; 5, 9; 6, 9; 7, 7; 8, 5; 9, 3; 10, 2; 11, 1];
% prob_dict(:,2) = prob_dict(:,2)/63;
% expected_rew_analytic = 0;
% expected_rew_analytic_conditional = zeros(length(prob_dict),length(prob_dict));
% expected_minrew_analytic = 0;
% expected_minrew_analytic_conditional = zeros(length(prob_dict),length(prob_dict));
% 
% for j = 1: length(prob_dict)
%     for k = 1: length(prob_dict)
%         expected_rew_analytic =                  prob_dict(j,2) * prob_dict(k,2) * gamma^max(j,k) * max_rew + expected_rew_analytic;
%         expected_rew_analytic_conditional(j,k) = prob_dict(j,2) * prob_dict(k,2) * gamma^max(j,k) * max_rew;    
%         
%         if j~=k
%             expected_minrew_analytic =                  prob_dict(j,2) * prob_dict(k,2) * gamma^min(j,k) * min_rew + expected_minrew_analytic;
%             expected_minrew_analytic_conditional(j,k) = prob_dict(j,2) * prob_dict(k,2) * gamma^min(j,k) * min_rew;    
%         else
%             expected_minrew_analytic =                  prob_dict(j,2) * prob_dict(k,2) * gamma^min(j,k) * max_rew + expected_minrew_analytic;
%             expected_minrew_analytic_conditional(j,k) = prob_dict(j,2) * prob_dict(k,2) * gamma^min(j,k) * max_rew;    
%     
%         end
%     end
% end



% 
% %What if we just finish the episode as soon as we can? (Greedy no com
% %policy GNCP):
% h_min = histogram(d_min);
% [count2, edges2] = histcounts(d_min);
% prob2 = count2/NI;
% 
% % computing the expeted reward
% range2 = length(prob2);
% discount_vect2 = ones(1,range2);
% for i = 1:range
%     discount_vect2(i) = gamma^(i);
% end
% rew_vect = discount_vect * max_rew;
% expected_rew = rew_vect * transpose(prob);



















function goal_coord = num_to_coord(goal_num, n)
%We compute coordinates of the goal point:
%inputs:

%  goal_num: put the location number of any point (not necessarily a goal
%  point) here!

%  n: put the width/length of the grid world here, it is assumed that the
%  gird world is square
    for i = 1:n
        if  goal_num <= (i * n -1) 
            goal_coord(1) = i;
            goal_coord(2) = goal_num - ((i-1) * n -1);
            break
        end
    end
end



%We compute the maximum distance of the agents from the goal point at each
%single iteration
function [max_dist,min_dist] = compute_distance(goal_coord, agents_loc)
    [~,no_of_agents,~] = size(agents_loc);
    dist = zeros(no_of_agents, 1);
    
    for i = 1: no_of_agents
        dist(i) = abs(goal_coord(1) - agents_loc(:,i,1)) + abs(goal_coord(2) - agents_loc(:,i,2));
    end
    
    max_dist = max(dist);
    min_dist = min(dist);
    
end


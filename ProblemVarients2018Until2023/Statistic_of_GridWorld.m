
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Understanding the statistic of the gridworld problem%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
close all
clc


%% Initialization of control parameters
N = 4; % the width and the length of the gridworld
G = 16; % goal point location
NI = 10^6; % Number of Iterations

%% Non controlable paramters
K = 2; % number of agents

%% Zero Initialization
l = zeros(NI, K, 2); %Agent coordinates at all iterations
d = zeros(NI, 1);
counter = 0;



%% Experimenting
%We compute coordinates of the goal point:
G_coord = num_to_coord(G, N);

%Generate a set that does not include the first coordinate of the goal point
g_comp_set_1 = 1: 1 : N;
g_comp_set_1(G_coord(1)) = [];

%Generate a set that does not include the second coordinate of the goal point
g_comp_set_2 = 1: 1 : N;
g_comp_set_2(G_coord(2)) = [];



%We generate the random coordinates of the K agents in the gridworld for NS number of times:
for i = 1:NI
    %generate the coordinates of K=2 agents at iteration N
    temp = randi(N, K, 2);

    %make sure no location is equal to the goal location
    if temp(1,:) == G_coord
        temp(1,1) = randsample(g_comp_set_1,1);
        temp(1,2) = randsample(g_comp_set_2,1);
    elseif temp(2,: ) == G_coord
        temp(2,1) = randsample(g_comp_set_1,1);
        temp(2,2) = randsample(g_comp_set_2,1);
        disp(temp(2,: ))
    end 
    
    if temp(1,:) == G_coord
        counter =  counter + 1;
    elseif temp(2,: ) == G_coord
        counter =  counter + 1;
    end
    l(i,:,:) = temp;    
end



%We compute the maximum distance of the agents from the goal point at each
%single iteration
for i = 1:NI
    d(i) = compute_distance(G_coord, l(i,:,:));
end

h = histogram(d);
[count, edges] = histcounts(d);



























%We compute coordinates of the goal point:
function goal_coord = num_to_coord(goal_num, n)
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
function max_dist = compute_distance(goal_coord, agents_loc)
    [~,no_of_agents,~] = size(agents_loc);
    dist = zeros(no_of_agents, 1);
    
    for i = 1: no_of_agents
        dist(i) = sum(abs(goal_coord - agents_loc(:,:,i)));
    end
    
    max_dist = max(dist);
    
end


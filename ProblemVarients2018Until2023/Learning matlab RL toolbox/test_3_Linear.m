%Test reinforcement learning matlab toolbox
%Codes are implemented from https://nl.mathworks.com/help/reinforcement-learning/ug/train-q-learning-agent-to-solve-basic-grid-world.html
%and from https://nl.mathworks.com/help/reinforcement-learning/ref/rldqnagent.html
clear
clc


%Create the basic grid world environment.
env = rlPredefinedEnv("BasicGridWorld");

%Obtaining the state-action space size:
actInfo = getActionInfo(env);
act_size = size(actInfo.Elements);

obsInfo = getObservationInfo(env);
obs_size = size(obsInfo.Elements);

%initializing the
env.ResetFcn = @() 2;

%Fix the random generator seed for reproducibility
rng(0)

%Creating a deep Q agent
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
%Critic Representation:
% statePath = [                                         %a layer graph is designed to take state inputs
%     %imageInputLayer([4 1 1], 'Normalization', 'none', 'Name', 'state')
%     sequenceInputLayer([1],'Name', 'state')
%     fullyConnectedLayer(24, 'Name', 'CriticStateFC1') %what is Output size
%     reluLayer('Name', 'CriticRelu1')
%     fullyConnectedLayer(24, 'Name', 'CriticStateFC2')];
% actionPath = [
%     sequenceInputLayer([1],'Name', 'action')
%     fullyConnectedLayer(24, 'Name', 'CriticActionFC1')]; %why no relu?
% commonPath = [
%     additionLayer(2,'Name', 'add') %two is the number of inputs: in1 and in2 (the names are set by defult)
%     reluLayer('Name','CriticCommonRelu')
%     fullyConnectedLayer(1, 'Name', 'output')];        %the number of outputs is 1

% criticNetwork = layerGraph(statePath);
% criticNetwork = addLayers(criticNetwork, actionPath);
% criticNetwork = addLayers(criticNetwork, commonPath);    %addLayers just tells which layers are in the network
% criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1'); %connectLayers indicates how layers are exactly connected
% criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
% plot(criticNetwork) %what is add in the critic network?
criticOpts = rlRepresentationOptions('LearnRate',0.01,'GradientThreshold',1);
critic = rlRepresentation(basisFcn,W0,obsInfo,actInfo);

%Configuring options and creating the DQN
agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN',false, ...    
    'TargetUpdateMethod',"periodic", ...
    'TargetUpdateFrequency',4, ...   
    'ExperienceBufferLength',100000, ...
    'DiscountFactor',0.90, ...
    'MiniBatchSize',256);
agent = rlDQNAgent(critic,agentOpts);

%To train the agent, first specify the training options. For this example, use the following options:
%** Train for at most 200 episodes, with each episode lasting at most 50 time steps.
%** Stop training when the agent receives an average cumulative reward greater than 10 over 30 consecutive episodes.
trainOpts = rlTrainingOptions;
%trainOpts.MaxStepsPerEpisode = 50;
%trainOpts.MaxEpisodes= 20000;
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 11;
trainOpts.ScoreAveragingWindowLength = 30;

%Training the agent - If you want to do training, doTraining should be
%equal to ture
doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load pretrained agent for the example.
    load('basicGWQAgent.mat','qAgent')
end

%Visualization
plot(env)
env.Model.Viewer.ShowTrace = true;
env.Model.Viewer.clearTrace;

%Simulate the agent
sim(agent,env)
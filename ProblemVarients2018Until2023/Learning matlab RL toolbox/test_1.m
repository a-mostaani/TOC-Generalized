%Test reinforcement learning matlab toolbox
%Codes are implemented from https://nl.mathworks.com/help/reinforcement-learning/ug/train-q-learning-agent-to-solve-basic-grid-world.html


%Create the basic grid world environment.
env = rlPredefinedEnv("BasicGridWorld");

%initializing the
env.ResetFcn = @() 2;

%Fix the random generator seed for reproducibility
rng(0)

%Creating Q-Learning agent
qTable = rlTable(getObservationInfo(env),getActionInfo(env));
tableRep = rlRepresentation(qTable);
tableRep.Options.LearnRate = 1;

%Configuring options
agentOpts = rlQAgentOptions;
agentOpts.EpsilonGreedyExploration.Epsilon = .04;
qAgent = rlQAgent(tableRep,agentOpts);

%To train the agent, first specify the training options. For this example, use the following options:
%** Train for at most 200 episodes, with each episode lasting at most 50 time steps.
%** Stop training when the agent receives an average cumulative reward greater than 10 over 30 consecutive episodes.
trainOpts = rlTrainingOptions;
trainOpts.MaxStepsPerEpisode = 50;
trainOpts.MaxEpisodes= 200;
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 11;
trainOpts.ScoreAveragingWindowLength = 30;

%Training the agent - If you want to do training, doTraining should be
%equal to ture
doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(qAgent,env,trainOpts);
else
    % Load pretrained agent for the example.
    load('basicGWQAgent.mat','qAgent')
end

%Visualization
plot(env)
env.Model.Viewer.ShowTrace = true;
env.Model.Viewer.clearTrace;

%Simulate the agent
sim(qAgent,env)
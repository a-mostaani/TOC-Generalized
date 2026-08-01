% DQN Network
% Note: This code assumes that you have already set up the necessary environment and have the required libraries installed.

% Import necessary libraries
import rl.*
import rl.representation.*
import rl.action.*
import rl.memory.*
import rl.learn.*
import rl.util.*

% Set the seed for reproducibility (optional)
rng(0)

% Define the DQN network architecture
stateSize = 4; % Define the size of the state input
actionSize = 2; % Define the size of the action space
layers = [
    imageInputLayer([stateSize 1 1]) % Input layer for the state
    fullyConnectedLayer(24) % Fully connected layer with 24 neurons
    reluLayer % ReLU activation function
    fullyConnectedLayer(24) % Another fully connected layer with 24 neurons
    reluLayer % ReLU activation function
    fullyConnectedLayer(actionSize) % Output layer with neurons equal to the action space size
    ];

% Define the options for the DQN agent
agentOpts = rlDQNAgentOptions;
agentOpts.EpsilonGreedyExploration.Epsilon = 0.1; % Exploration rate (set your desired value)
agentOpts.ExperienceBufferLength = 10000; % Experience replay buffer length
agentOpts.TargetUpdateMethod = "periodic"; % Update target network periodically
agentOpts.TargetUpdateFrequency = 4; % Frequency of target network update
agentOpts.DiscountFactor = 0.99; % Discount factor for future rewards
agentOpts.MiniBatchSize = 32; % Mini-batch size for experience replay
agentOpts.SaveExperienceBufferWithAgent = true; % Save the replay buffer with the agent

% Create the DQN agent
dqnAgent = rlDQNAgent(layers, agentOpts);

% Define the training options
trainOpts = rlTrainingOptions;
trainOpts.MaxStepsPerEpisode = 1000; % Maximum number of steps per episode (set your desired value)
trainOpts.MaxEpisodes = 100; % Maximum number of training episodes (set your desired value)
trainOpts.StopTrainingCriteria = "AverageReward"; % Stop training when the average reward exceeds a threshold
trainOpts.StopTrainingValue = 200; % Average reward threshold for stopping training
trainOpts.ScoreAveragingWindowLength = 20; % Window length for averaging the rewards

% Train the DQN agent
trainingStats = train(dqnAgent, env, trainOpts);

% Plot the training results
plot(trainingStats.EpisodeIndex, trainingStats.Score)
xlabel('Episode')
ylabel('Score')
title('Training Performance')
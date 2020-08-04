import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from noise import OUNoise

# Parameters for continuous control are taken from the original paper on DDPG: Continuous Control with Deep Reinforcement Learning, https://arxiv.org/pdf/1509.02971.pdf
#Using Batch normalization, you can learn effectively across many different tasks with differing types of units. <-- According to the paper.
#Moreover, it maintains a running average of the mean and variance to use for normalization during testing (exploration or evaluation).

LEARNING_RATE = 1e-3 #critic learning rate
GAMMA = 0.99 #discount factor
WEIGHT_DECAY = 0.01 #L2 weight decay 
TAU = 0.001 #soft target update


class Critic(nn.Module):
    """Value approximator V(pi) as Q(s, a|θ)"""
    def __init__(self, state_size, action_size, fc1=200, fc2=300):
        """
        @Param:
        1. state_size: number of observations, i.e. env.observation_space.shape[0] 
        2. action_size: number of actions, i.e. env.action_space.shape[0]
        3. fc1: number of hidden units in the first fully connected layer. Default = 200.
        4. fc2: number of hidden units in the second fully connected layer, default = 300.
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1 + action_size, fc2) #the reasons why we're adding fc1 + action_size is because we need to map (staate, action) -> Q-values. 
        self.fc3 = nn.Linear(fc2, 1) #Q-value
        #Batch Normalization
        self.bn1 = nn.BatchNorm1d(fc1)
    
    def reset_parameters(self):
        pass
    def forward(self, state, action):
        pass

        
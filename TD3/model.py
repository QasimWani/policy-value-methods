import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#Twin Delayed Deep Deterministic Policy Gradient (TD3) implementation for continuous action space control.
#Paper: https://arxiv.org/pdf/1802.09477.pdf
#Author: https://github.com/sfujim/TD3

class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action, fc1=256, fc2=256):
        """
        Initializes actor object.
        @Param:
        1. state_size: env.observation_space.shape[0].
        2. action_size: env.action_space.shape[0].
        3. max_action: abs(env.action_space.low), sets boundary/clip for policy approximation.
        4. fc1: number of hidden units for the first fully connected layer, fc1. Default = 256.
        5. fc1: number of hidden units for the second fully connected layer, fc1. Default = 256.
        """
        super(Actor, self).__init__()

        #Layer 1
        self.fc1 = nn.Linear(state_size, fc1)
        #Layer 2
        self.fc2 = nn.Linear(fc1, fc2)
        #Layer 3
        self.mu = nn.Linear(fc2, action_size)

        #Define boundary for action space.
        self.max_action = max_action
    
    def forward(self, state):
        """Peforms forward pass to map state--> pi(s)"""
        #Layer 1
        x = self.fc1(state)
        x = F.relu(x)
        #Layer 2
        x = self.fc2(x)
        x = F.relu(x)
        #Output layer
        x = self.mu(x)
        mu = torch.tanh(mu)#set action b/w -1 and +1
        return self.max_action * mu
        





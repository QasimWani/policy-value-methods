import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#Using Batch normalization, you can learn effectively across many different tasks with differing types of units. <-- According to the paper.
#Moreover, it maintains a running average of the mean and variance to use for normalization during testing (exploration or evaluation).

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
        #Layer 1
        self.fc1 = nn.Linear(state_size, fc1)
        self.bn1 = nn.BatchNorm1d(fc1)
        #Layer 2
        self.fc2 = nn.Linear(fc1 + action_size, fc2) #the reasons why we're adding fc1 + action_size is because we need to map (state, action) -> Q-values. 
        self.bn2 = nn.BatchNorm1d(fc2)
        #Output layer
        self.q = nn.Linear(fc2, 1) #Q-value
        
        #Initialize Weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters by setting a noise from distribution following from its respective hidden unit size.
        Format for (-fx,fx) followed from the original paper.
        """
        
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)
        
    def forward(self, state, action):
        """
        Performs a single forward pass to map (state,action) to Q-value
        @Param:
        1. state: current observations, shape: (env.observation_space.shape[0],)
        2. action: immediate action to evaluate against, shape: (env.action_space.shape[0],)
        @Return:
        - q-value
        """
        #Layer #1
        x_state = self.fc1(state) #state_space -> fc1=200
        x_state = self.bn1(x_state)
        x_state = F.relu(x_state)
        
        #Layer #2
        x = torch.cat((x_state, action), dim=1) #Concatenate state with action. Note that the specific way of passing x_state into layer #2.
        x = self.fc2(x) #fc1=200 + action_space --> fc2=300
        x = self.bn2(x)
        x = F.relu(x)

        #Output
        value = self.q(x) #fc2=300 --> 1
        return value
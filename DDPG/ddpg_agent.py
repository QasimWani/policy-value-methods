import numpy as numpy
import copy
import random
from collections import namedtuple, deque

from actor import Actor
from critic import Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

# Parameters for continuous control are taken from the original paper on DDPG: Continuous Control with Deep Reinforcement Learning, https://arxiv.org/pdf/1509.02971.pdf
LR_CRITIC = 1e-3 #critic learning rate
LR_ACTOR = 1e-4 #actor learning rate
GAMMA = 0.99 #discount factor
WEIGHT_DECAY = 0.01 #L2 weight decay 
TAU = 0.001 #soft target update
BUFFER_SIZE = 1e6 #replay buffer size
MINI_BATCH = 64 #minibatch size



class ReplayBuffer():
    """
    Implementation of a fixed size replay buffer as used in DQN algorithms.
    The goal of a replay buffer is to unserialize relationships between sequential experiences, gaining a better temporal understanding.
    """
    def __init__(self, action_size, buffer_size=BUFFER_SIZE, batch_size=MINI_BATCH):
        """
        Initializes the buffer.
        @Param:
        1. action_size: env.action_space.shape[0]
        2. buffer_size: Maximum length of the buffer for extrapolating all experiences into trajectories. default - 1e6 (Source: DeepMind)
        3. batch_size: size of mini-batch to train on. default = 64.
        """
        self.action_size = action_size
        self.replay_memory = deque(maxlen=buffer_size) #Experience replay memory object
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"]) #standard S,A,R,S',done
        
    def add(self, state, action, reward, next_state, done):
        """Adds an experience to existing memory"""
        trajectory = self.experience(state, action, reward, next_state, done)
        self.replay_memory.append(trajectory)
    
    def sample(self):
        """Randomly picks minibatches within the replay_buffer of size mini_batch"""
        experiences = random.sample(self.replay_memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):#override default __len__ operator
        """Return the current size of internal memory."""
        return len(self.memory)
        




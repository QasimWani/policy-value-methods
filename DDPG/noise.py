import numpy as np
import copy
import random

#This class defines the OUNoise structure taken from Physics used originally for modelling the velocity of a Brownian particle.
#We are using this to setup our noise because it follows the 3 conditions of MDP process and is Gaussian process.
#Read more about Ornstein-Uhlenbeck process at: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

#Parameters for theta and sigma taken from Contionous Control for Deeep Reinforcement Learning.

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
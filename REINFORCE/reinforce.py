# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
### Implement REINFORCE (Monte Carlo Policy Gradients) to solve LunarLander task in OpenAI-gym


# %%
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd

import torch
# torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical #bernoulli distribution

import time #human agent


# %%
env_id = 'LunarLander-v2'
env = gym.make(env_id)
# env.seed(0)


# %%
env.observation_space


# %%
env.action_space


# %%
## check for bounds
env.observation_space.is_bounded()


# %%
LEARNING_RATE = 0.5*1e-2 #set learning rate

# %% [markdown]
# ## Define Policy

# %%
class Policy(nn.Module):
    """Defines the general policy for an agent following simple NN architecture"""
    def __init__(self, state_size, action_size, h1=16):
        """Creates the model using a 3 Hidden layer NN"""
        super(Policy, self).__init__()#inherit methods from parent class & override forward f(x)
        self.fc1 = nn.Linear(in_features=state_size, out_features=h1)
        self.fc2 = nn.Linear(in_features=h1, out_features=action_size)
    def forward(self, x):
        """
        Performs one-pass from state -> action mapping.
        @Param:
        1. x - input state
        @return:
        x - action as a set of vector following stochastic measure. softmax output to logits from NN.
        """
        if(type(x) != torch.Tensor):
            try:
                x = torch.from_numpy(x).float().unsqueeze(0)#convert ndarray to torch.Tensor object
            except:
                raise TypeError(f"expected type torch.Tensor. got {type(x)}")
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    def act(self, state):
        """
        Uses current deterministic policy to determine the set of action to perform
        @param:
        1. state: input state of env. shape = env.observation_space.shape[0]
        @return:
        - action: (int) discrete action to take by the agent.
        - log_probs: (array_like) log of output from softmax unit. set of log probabilities.
        """
        probs = self.forward(state).cpu() #get estimated action following stochastic measure
        m = Categorical(probs)#get Bernoulli distribution of action
        action = m.sample() #returns the action based on the probability of each based on Benoulli(probs)
        return action.item(), m.log_prob(action)


# %%
policy = Policy(env.observation_space.shape[0], env.action_space.n)


# %%
### Training mode
policy.load_state_dict(torch.load("model.pth"))


# %%
optimizer = optim.Adam(params=policy.parameters(), lr=LEARNING_RATE)

# %% [markdown]
# ### Train using REINFORCE

# %%
def reinforce(num_episode=6000, max_tau=1000, gamma=1.0, print_every=100):
    """
    Implements the Reinforce algorithm.
    See paper for more details: https://bit.ly/REINFORCE_paper
    @param:
    1. num_episode: number of epochs to train for.
    2. max_tau: length of trajectory, 𝝉.
    3. gamma: discounted return, γ.
    4. print_every: pprint details after very X epochs.
    @return:
    - scores: (array_like) expected return over epochs.
    """
    scores_deque = deque(maxlen=100)# ∑R for last N=100 episodes
    scores = []
    for i_episode in range(1, num_episode + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()#reset the environment at the start of each episode
        for t in range(max_tau):#iterate through trajectory
            action, log_probs = policy.act(state)
            state, reward, done,_ = env.step(action)
            rewards.append(reward)
            saved_log_probs.append(log_probs)
            if(done):
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        #calculate Reward with gamma in account
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([g*r for g,r in zip(discounts, rewards)])#R = sum(γ^0*reward_0 + γ^1*reward_1 + γ^n*reward_n)

        ### Implement Stochastic Gradient Ascent
        policy_loss = []#estimated loss of the Policy (should be maximized towards optimal policy, see Hill climb)
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob*R)#-ve takes in account for gradient ascent
        
        policy_loss = torch.cat(policy_loss).sum() #find total loss, U(Θ)
        
        optimizer.zero_grad()#clear gradients
        policy_loss.backward()#performs back-prop
        optimizer.step()#performs a single update
    
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque)>=200.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break
    return scores


# %%
scores = reinforce()

# %% [markdown]
# # Solved Environment in 5299 Episodes!
# %% [markdown]
# #### Construct plot with moving averages

# %%
df = pd.DataFrame(scores)


# %%
plt.plot(df, label="Original")
plt.plot(df.rolling(window=50).mean(), label="Rolling")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()


# %%
reward_arr = []
for i in range(6):
    total_reward = 0
    state = env.reset()
    while True:
        action, _ = policy.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            reward_arr.append(total_reward)
            break 


# %%
np.mean(reward_arr)

# %% [markdown]
# ### Save the model weights

# %%
torch.save(policy.state_dict(), "model.pth")


# %%
load_weights = torch.load("model.pth")


# %%
policy.load_state_dict(load_weights)



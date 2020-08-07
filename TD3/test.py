import numpy as np
import torch
import gym
import random
import matplotlib.pyplot as plt

import utils
from TD3 import Agent

env_id = "BipedalWalker-v3"
env = gym.make(env_id)


#Set exploration noise for calculating action based on some noise factor
exploration_noise = 0.1

#Define observation and action space
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

#Create Agent
policy = Agent(state_space, action_space, max_action)
try:
    policy.load("final")
except:
    raise IOError("Couldn't load policy")


#Train the model
max_episodes = 15
max_timesteps = 2000

ep_reward = [] #get list of reward for range(max_episodes)

for episode in range(1, max_episodes+1):
    avg_reward = 0
    state = env.reset()
    for t in range(1, max_timesteps + 1):
        # select action and add exploration noise:
        action = policy.select_action(state) + np.random.normal(0, max_action * exploration_noise, size=action_space)
        action = action.clip(env.action_space.low, env.action_space.high)
            
        # take action in env:
        next_state, reward, done, _ = env.step(action)
        state = next_state
            
        avg_reward += reward

        #Renders an episode
        env.render()


        # if episode is done then display stats:
        if(done or t >=max_timesteps):
            print(f"Episode {episode} reward: {avg_reward} | Rolling average: {np.mean(ep_reward)}")
            print(f"Current time step: {t}")
            ep_reward.append(avg_reward)
            break 
    

env.close()

#Display Scores
fig = plt.figure()
plt.plot(np.arange(1, len(ep_reward) + 1), ep_reward)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
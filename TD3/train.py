import numpy as np
import torch
import gym
import random
import matplotlib.pyplot as plt

import utils
from TD3 import Agent

env_id = "BipedalWalker-v3"
env = gym.make(env_id)


#set seeds
env.seed(0)
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


#Set exploration noise for calculating action based on some noise factor
exploration_noise = 0.1

#Define observation and action space
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

#Create Agent
policy = Agent(state_space, action_space, max_action)

try:
    policy.load("00")
except:
    raise IOError("Couldn't load policy")

#Create Replay Buffer
replay_buffer = utils.ReplayBuffer()


#Train the model
max_episodes = 100
max_timesteps = 2000

ep_reward = [] #get list of reward for range(max_episodes)

for episode in range(1, max_episodes+1):
    avg_reward = 0
    state = env.reset()
    for t in range(1, max_timesteps + 1):
        # select action and add exploration noise:
        action = policy.select_action(state)
        action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
        action = action.clip(env.action_space.low, env.action_space.high)
            
        # take action in env:
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
            
        avg_reward += reward

        #Renders an episode
        # env.render()


        # if episode is done then update policy:
        if(done or t >=max_timesteps):
            print(f"Episode {episode} reward: {avg_reward} | Rolling average: {np.mean(ep_reward)}")
            print(f"Current time step: {t}")
            policy.train(replay_buffer) #training mode
            ep_reward.append(avg_reward)
            break 
    
    if(episode % 100 == 0):
        #Save policy and optimizer every 100 episodes
        policy.save(str("%02d" % (episode//100)))

#Display Scores
fig = plt.figure()
plt.plot(np.arange(1, len(ep_reward) + 1), ep_reward)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
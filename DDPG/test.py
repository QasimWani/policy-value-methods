import gym
import torch
import numpy as np
from ddpg_agent import Agent
import matplotlib.pyplot as plt

env = gym.make('BipedalWalker-v3')

state_dim = int(env.observation_space.shape[0])
action_dim = int(env.action_space.shape[0])
agent = Agent(state_size=state_dim, action_size=action_dim)


def ddpg(episodes, step, pretrained=True, noise=False):

    if pretrained:
        agent.actor_local.load_state_dict(torch.load('./models/weights/checkpoint_actor.pth', map_location="cpu"))
        agent.critic_local.load_state_dict(torch.load('./models/weights/checkpoint_critic.pth', map_location="cpu"))
        agent.actor_target.load_state_dict(torch.load('./models/weights/checkpoint_actor.pth', map_location="cpu"))
        agent.critic_target.load_state_dict(torch.load('./models/weights/checkpoint_critic.pth', map_location="cpu"))

    reward_list = []
    time_list = []

    for i in range(episodes):

        state = env.reset()
        score = 0

        for t in range(step):

            env.render()

            action = agent.act(state, add_noise=noise)
            next_state, reward, done, info = env.step(action[0])
            state = next_state.squeeze()
            score += reward

            if done:
                print('Reward: {} | Episode: {}/{}'.format(score, i, episodes))
                print(f"Timesteps: {t}. Time (sec): {format(t/50, '.3f')}") #fps according to OpenAI = 50
                break

        reward_list.append(score)
        time_list.append(t)

    print('Training saved')
    return reward_list, time_list


scores, time_list = ddpg(episodes=5, step=2000)

#Display Scores
fig = plt.figure()
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

#Display time
fig = plt.figure()
plt.plot(np.arange(1, len(time_list) + 1), time_list)
plt.ylabel('Time')
plt.xlabel('Episode #')
plt.show()
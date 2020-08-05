import gym
import torch
import numpy as np
from ddpg_agent import Agent
import matplotlib.pyplot as plt

env = gym.make('BipedalWalker-v3')

state_dim = int(env.observation_space.shape[0])
action_dim = int(env.action_space.shape[0])
agent = Agent(state_size=state_dim, action_size=action_dim)


def ddpg(episodes, step, pretrained=False, noise=False):

    if pretrained:
        agent.actor_local.load_state_dict(torch.load('models/2/checkpoint_actor.pth', map_location="cpu"))
        agent.critic_local.load_state_dict(torch.load('models/2/checkpoint_critic.pth', map_location="cpu"))
        agent.actor_target.load_state_dict(torch.load('models/2/checkpoint_actor.pth', map_location="cpu"))
        agent.critic_target.load_state_dict(torch.load('models/2/checkpoint_critic.pth', map_location="cpu"))

    reward_list = []

    for i in range(episodes):

        state = env.reset()
        score = 0

        for t in range(step):

            # env.render()

            action = agent.act(state, add_noise=True)
            next_state, reward, done, info = env.step(action[0])
            agent.step(state, action, reward, next_state, done)
            state = next_state.squeeze()
            score += reward

            if done:
                #print recent statistics every 10 episodes
                if(i%10):
                    print('Reward: {} | Episode: {}/{}'.format(score, i, episodes))
                    print(f"Timesteps: {t}. Time (sec): {format(t/50, '.3f')}") #fps according to OpenAI = 50
                break

        reward_list.append(score)

        #Save model every 100 episodes
        if(i%100):
            print(f"\nMEAN REWARD: {np.mean(reward_list)}\n")
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_'+str("%03d" % (i//100))+'.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_'+str("%03d" % (i//100))+'.pth')
            torch.save(agent.actor_target.state_dict(), 'checkpoint_actor_t_'+str("%03d" % (i//100))+'.pth')
            torch.save(agent.critic_target.state_dict(), 'checkpoint_critic_t_'+str("%03d" % (i//100))+'.pth')
            
        if score >= 270:
            print('Task Solved')
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            torch.save(agent.actor_target.state_dict(), 'checkpoint_actor_t.pth')
            torch.save(agent.critic_target.state_dict(), 'checkpoint_critic_t.pth')
            break
        

    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
    torch.save(agent.actor_target.state_dict(), 'checkpoint_actor_t.pth')
    torch.save(agent.critic_target.state_dict(), 'checkpoint_critic_t.pth')

    print('Training saved')
    return reward_list


scores = ddpg(episodes=10, step=2000, pretrained=False, noise=True)

fig = plt.figure()
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
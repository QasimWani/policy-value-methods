#!/usr/bin/env python
# coding: utf-8

# ### Use the Asynchronous Advantage Actor Critic (A3C) Policy Gradient Method to solve Breakout

# In[73]:


import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import progressbar as pb
from scipy.signal import lfilter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from parallelEnv import parallelEnv
from PIL import Image


# In[2]:


env_id = 'Breakout-v4'
env = gym.make(env_id)


# In[3]:


env.observation_space


# In[4]:


print(f"Actions:{env.action_space}\nMeanings:{env.unwrapped.get_action_meanings()}")


# In[5]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #convert to GPU if available


# ## Preprocess Image

# In[148]:


## Utils
def preprocess_single_frame(image, bkg_color = np.array([144, 72, 17]), n=16):
    """
    Converts an image from RGB channel to B&W channels.
    Also performs downscale to 80x80. Performs normalization.
    @Param:
    1. image: (array_like) input image. shape = (210, 160, 3)
    2. bkg_color: (np.array) standard encoding for brown in RGB with alpha = 0.0
    @Return:
    - img: (array_like) B&W, downscaled, normalized image of shape (80x80)
    """
    img = np.mean(image[35:195:2,::2]-bkg_color, axis=-1)/255.
    return img

#Utils
def preprocess_batch(images, bkg_color = np.array([144, 72, 17])):
    """
    convert outputs of parallelEnv to inputs to pytorch neural net"""
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop
    list_of_images_prepro = np.mean(list_of_images[:,:,34:-16:2,::2]-bkg_color,
                                    axis=-1)/255.
    batch_input = np.swapaxes(list_of_images_prepro,0,1)
    return torch.from_numpy(batch_input).float().to(device)


# In[119]:


state = env.reset()
for _ in range(20):#skip 20 frames
    frame, _, _, _ = env.step(np.random.randint(0, env.action_space.n))


# In[121]:


#Plot processed and raw image
plt.subplot(1,2,1)
plt.imshow(frame)
plt.title('original image')

plt.subplot(1,2,2)
plt.title('preprocessed image')
# 80 x 80 black and white image
plt.imshow(preprocess_single_frame(frame)[0], cmap='Greys')
plt.show()


# # Actor-Critic

# ![actor critic achitecture](https://www.mdpi.com/energies/energies-09-00725/article_deploy/html/images/energies-09-00725-g001-1024.png)

# In[122]:


#DEFINE Constants
GAMMA = 0.99
TAU = 1.0
LR = 0.5*1e-4
MAX_EPISODE_LENGTH = 1e8
BETA = 0.01 #entropy coefficient


# In[123]:


class ActorCritic(nn.Module):
    def __init__(self, action_size=4, num_frames=2):
        super(ActorCritic, self).__init__()
        self.action_size = action_size
        
        #Define the CNN for Actor & Critic
        self.conv1 = nn.Conv2d(num_frames, 32, 3, stride=2, padding=1) #output = 40x40x32
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1) #output = 20x20x32
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1) #output = 10x10x32
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1) #output = 5x5x32
        
        self.size = 5*5*32 #800

        #FC layer
        self.lstm = nn.LSTMCell(input_size=self.size, hidden_size=256)#lstm cell to prevent vanishing gradients
        
        # Define Actor and Critic network
        # Critic evaluates the state value function, V(π) using TD estimate.
        # Actor evaluates the policy π(a|s) distribution
        
        self.critic_linear, self.actor_linear = nn.Linear(256, 1), nn.Linear(256, self.action_size)
        
    def forward(self, x, hx, cx):
        """
        Peforms one-pass for the Conv layers.
        @Param:
        1. x - shape: (2, 80, 80); 2 stacked frames of 80x80 images
        2. hx - hidden state of the RNN. shape: (1x256)
        3. cx - confidence state of the RNN. shape: (1x256)
        @Return:
        1. critic estimated value, V(π)
        2. actor policy distribution, π(a|s) as logits
        """
        #4 conv nets without max pool layers, simple Relu activation f(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1,self.size) #flatten
        hx, cx = self.lstm(x, (hx, cx)) #dynamic calculation for final confidence & hidden state
        value = self.critic_linear(hx) #CRITIC: calculates estimated state value function, V(π)
        logits = self.actor_linear(hx) #ACTOR:  calculates policy distribution π(a|s)
        
        return logits, value, hx, cx


# In[124]:


main_model = ActorCritic(num_frames=1)


# In[125]:


main_model


# # Optimizer
# <p> A critical component to an A3C model is the ability to share parameters across
#     multiple agents running asynchronously such that they can collectively learn from
#     each other. This is done by the cross-integration (sharing) of gradients across all processes.
# </p>

# In[126]:


class SharedOptimizer(optim.Adam):
    """Implementation of shared parameter model using Adam optimizer"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedOptimizer, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)


# In[127]:


optimizer = SharedOptimizer(main_model.parameters()) #define optimizer (uses Adam, instead of SGD)


# ## Generalized Advantage Estimator 
# <br>
# <p>
#     Generalized Advantage Estimator (GAE) helps us pick the best value for N-step boostrapping
#     by incorporating λ as an added hyper-parameter to tune accordingly that will minimize the
#     bias-variance tradeoff.
#     <br>
#     The derivation can be thought of as <a href="https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/#the-generalized-advantage-estimator">the exponentially-decayed sum of residual terms.</a>
# </p>
# <p><strong>See the following <a href="https://arxiv.org/pdf/1506.02438.pdf">derivation</a> for GAE estimator:</strong></p>

# ![GAE Derivation](https://res.cloudinary.com/crammer/image/upload/v1596251771/Screen_Shot_2020-07-31_at_11.15.47_PM_feuhld.png)

# In[128]:


discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # computes discounted reward


# In[129]:


def compute_cost(values, log_probs, actions, rewards, n=8):
    """
    Calculates the policy (actor) and value (critic) loss
    @Param:
    1. values: (tensor) list of V(s) estimator, critic.
    2. log_probs: (tensor) list of π(a|s) softmax output, actor.
    3. actions: (tensor) actions taken from rollout of trajectory.
    4. rewards: (tensor) rewards based on S,A pairs. true values, used to minimize loss.
    5. gae_lambda: (float) [0-1] value of lambda for residual calculation. used in N-step bootstrap.
    6. n: (int) number of parallel agents.
    @Return:
    - value_loss: (tensor) critic loss.
    - policy_loss: (tensor) actor loss.
    """
    np_values = values.view(-1).data.numpy() #convert torch.tensor to numpy array & flatten/reshape it
    #implement GAE
    rewards = torch.tensor(rewards).view(-1,1).squeeze(-1) # (20x8) --> (160)
    delta_t =  + GAMMA * (np_values[n:] - np_values[:-n])
    
    log_probs = log_probs.gather(1, torch.tensor(actions).view(-1,1))
    gae = discount(delta_t, GAMMA * TAU)#calculate Generative Advantage Estimate

    #calculate policy_loss = -log( π(a|s) ) * ( R - V(s) )
    policy_loss = -(log_probs.view(-1) * torch.FloatTensor(gae.copy())).sum()
    
    # l2 loss over value estimator
    rewards[-1] += GAMMA * np_values[-n]
    discounted_r = discount(np.asarray(rewards), GAMMA)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    
    value_loss = 0.5*(discounted_r - values[:-n,0]).pow(2).sum()

    entropy_loss = (-log_probs * torch.exp(log_probs)).sum() # entropy = ∑ -log(π(a|s))*e^(log(π(a|s)))
    return policy_loss + 0.5 * value_loss - BETA * entropy_loss
#     return policy_loss - value_loss -  BETA * entropy_loss #total loss


# ## Train

# In[130]:


# Utils
def sync_models(model, shared_model, i, pprint):
    """
    Syncs the gradients from local model to shared model as a critical part of the A3C algorithm.
    Updates the pointer based reference.
    @param:
    1. model: local model to sync from.
    2. shared_model: global model to sync into.
    3. i: episode number. Used in calculation of when to save the local model.
    4. pprint: model iteration to save, i.e. saves model when i%pprint = 0.
    """
    if(i%pprint == 0): #Save weights
        model_path = "models/model_" + str(i//pprint) + "_.pth"
        torch.save(model.state_dict(), model_path)
        
    for local_param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if(shared_param.grad is not None):
            return model, shared_model
        shared_param.grad = local_param.grad #sync
    return model, shared_model


# In[131]:


def get_stacked_frame(envs, n, reset=True):
    """
    Returns a 1x2x80x80 tensor by concatenating 2 processed frames.
    @Param:
    - reset: (boolean, optional) if True, gets the first 2 frames of the env.
    @return:
    - state: a 1x2x80x80 torch.Tensor
    - frame0: first frame of size 80x80 
    - frame1: second frame of size 80x80 
    """
    frame0 = envs.reset()
#     frame1,_,_,_ = envs.step([1]*n) #fire
    state = preprocess_batch([frame0]) #nx1x80x80 tensor
    return state, frame0


# In[41]:


def train(shared_model, envs, optimizer=None, num_episode=1000, num_steps=20, print_every=100):
    """
    Train A3C agent.
    @Param:
    1. shared_model: instance of ActorCritic class, globally shared model across all parallel agents.
    2. envs: parallel agents.
    3. optimizer: instance of SharedOptimizer class, default = None (created as local object).
    4. num_episode: (int) number of episodes to train for.
    5. num_steps: (int) number of forward pass to pass through. default = 20.
    6. print_every: (int) display statistics & save weights every n episodes. default = 100.
    @Return:
    - overall_reward: total reward per episode.
    - overall_cost: total cost based on the custom loss function over num_episode.
    - loss_deque: the N most recent loss values. default = 100.
    """
    # widget bar to display progress
    widget = ['training loop: ', pb.Percentage(), ' ', 
              pb.Bar(), ' ', pb.ETA() ]
    timer = pb.ProgressBar(widgets=widget, maxval=num_episode).start()
    
    # number of parallel instances
    n = len(envs.ps)
    
    model = ActorCritic(num_frames=1) #local model
    
    if(optimizer is None):#create optimizer
        optimizer = SharedOptimizer(shared_model.parameters(), lr=LR)
    
    model.train()#Set the local model in training mode
    
    #Extract stacked frames
    
    state, frame0 = get_stacked_frame(envs, n)# (nx2x80x80) Tensor
    episode_length = 0
    done = np.full(8, True)
    
    #return metrics
    overall_reward = []
    overall_cost = []
    loss_deque = deque(maxlen=100)
    score_deque = deque(maxlen=100)
    
    for i in range(1, num_episode+1):
        model.load_state_dict(shared_model.state_dict()) #syncs the shared model with the local model
            
        if done.any():
            cx, hx = torch.zeros(n, 256), torch.zeros(n, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()
        
        values = []; log_probs = []; rewards = []; actions = []
        for step in range(num_steps):
            episode_length += 1
            logits, value, hx, cx = model(state, hx, cx)
            prob = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            
            action = [torch.exp(log_prob[i]).multinomial(num_samples=1).item() for i in range(n)]
            frame0, reward, done, _ = envs.step(action)
            
            #----- Clip reward -------
            reward = np.clip(reward, -1, 1) # reward
            #-------------------------
            
#             frame1, r, done, _ = envs.step([0]*n)

            #stack frames to an nx2x80x80 Tensor
            state = preprocess_batch([frame0])
            
            #update data
            values.append(value.detach())
            log_probs.append(log_prob)
            rewards.append(reward)
            actions.append(action)
                        
            if done.any() or episode_length > MAX_EPISODE_LENGTH:
                episode_length = 0
                state, frame0 = get_stacked_frame(envs, n)#nx2x80x80 Tensor
                break
        overall_reward.append(sum(rewards)) #cumulitive reward per episode
        score_deque.append(overall_reward) #last 100 score
    
        #Solves broadcasting error
        next_value = torch.zeros(n,1) if done.any() else model(state, hx, cx)[1] #return value
        values.append(next_value.detach())

        #compute loss
        loss = compute_cost(torch.cat(values), torch.cat(log_probs), actions, rewards, n=n)
        overall_cost.append(loss.item())
        loss_deque.append(loss.item()) #last 100 losses
        if(i%print_every == 0):
            print("Episode: {0:d}, cost: {1:f}, score: {2:f}".format(i,np.mean(loss_deque), np.mean(score_deque)))
            print(overall_reward[-1])
        # update progress widget bar
        timer.update(i)
        
        optimizer.zero_grad()#reset gradient
        loss.backward() #perform backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1) #clip gradients
        #synchronize shared_model with local_model
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad # sync gradients with shared model

        #Perform single forward step
        optimizer.step()
    
    timer.finish()
    
    return overall_reward, overall_cost, loss_deque


# In[142]:


def train(envs, shared_model, shared_optimizer, num_episodes=1000, rnn_steps=20, print_every=10):
    # number of parallel instances
    n = len(envs.ps)
    
    model = ActorCritic(num_frames=1) # a local/unshared model
    state = torch.tensor(preprocess_single_frame(envs.reset(), n=n)) # get first state

    episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping

    for i in range(1, num_episodes + 1): # openai baselines uses 40M frames...we'll use 80M
        model.load_state_dict(shared_model.state_dict()) # sync with shared model

        hx = torch.zeros(n, 256) if done.any() else hx.detach()  # rnn activation vector
        cx = torch.zeros(n, 256) if done.any() else cx.detach()  # rnn activation vector
        values, logps, actions, rewards = [], [], [], [] # save values for computing gradientss

        for step in range(rnn_steps):
            episode_length += 1
            value, logit, hx = model((state.view(n,1,80,80), hx, cx))
            logp = F.log_softmax(logit, dim=-1)

            action = torch.exp(logp).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
            state, reward, done, _ = envs.step(action.numpy()[0])

            state = torch.tensor(preprocess_single_frame(state,n=n)) ; epr += reward
            reward = np.clip(reward, -1, 1) # reward
            done = done or episode_length >= 1e4 # don't playing one ep for too long
            
#             info['frames'].add_(1) ; num_frames = int(info['frames'].item())
#             if num_frames % 2e6 == 0: # save every 2M frames
#                 printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
#                 torch.save(shared_model.state_dict(), args.save_dir+'model.{:.0f}.tar'.format(num_frames/1e6))

#             if done: # update shared data
#                 info['episodes'] += 1
#                 interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
#                 info['run_epr'].mul_(1-interp).add_(interp * epr)
#                 info['run_loss'].mul_(1-interp).add_(interp * eploss)

#             if rank == 0 and time.time() - last_disp_time > 60: # print info ~ every minute
#                 elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
#                 printlog(args, 'time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
#                     .format(elapsed, info['episodes'].item(), num_frames/1e6,
#                     info['run_epr'].item(), info['run_loss'].item()))
#                 last_disp_time = time.time()

            if done.any(): # maybe print info.
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(preprocess_single_frame(env.reset(),n=n))

            values.append(value) ; logps.append(logp) ; actions.append(action) ; rewards.append(reward)
            if(i%print_every == 0):
                print(np.mean(logp))
                print(np.mean(reward))
                
        next_value = torch.zeros(n,1) if done else model((state.unsqueeze(0), hx, cx))[0]
        values.append(next_value.detach())

        loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        shared_optimizer.zero_grad() ; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad # sync gradients with shared model
        shared_optimizer.step()


# In[149]:


state = torch.tensor(preprocess_single_frame(envs.reset(), n=16)) # get first state


# In[150]:


state.shape


# In[ ]:


#FUCK ME MAN!!!


# ## Parallelization

# In[140]:


# load mulitple parallel agents, in this case 16.
envs = parallelEnv(env_id, n=16, seed=1234)


# In[141]:


train(envs, main_model, optimizer, num_episodes=100, print_every=5)


# In[22]:


plt.plot(overall_cost)
plt.xlabel("Episode #")
plt.ylabel("Overall cost")
plt.show()


# In[23]:


plt.plot(np.array(overall_reward).T[0])#reward for agent #1
plt.xlabel("Episode #")
plt.ylabel("Overall score for agent #1")
plt.show()


# In[24]:


def test():
    # state, frame0, frame1 = get_stacked_frame()
    frame0 = env.reset()
#     env.step(1) #fire
#     frame1 = env.step(0)[0]
    total_reward = 0 #reward per frame stacked

    # # perform nrand random steps in the beginning
    # for _ in range(5):
    #     frame1, reward1, is_done, _ = env.step(np.random.choice([RIGHT,LEFT]))
    #     frame2, reward2, is_done, _ = env.step(0)
    done = True
    actions = []
    for _ in range(1000):

        if done:
            cx, hx = torch.zeros(1, 256), torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        state = preprocess_batch([frame0])     

        logits, value, hx, cx = main_model(state, hx, cx)

        log_prob = F.softmax(logits, dim=-1)
        action = torch.exp(log_prob).multinomial(num_samples=1).item()
        frame0, r1, is_done, _ = env.step(action)


        actions.append(action)
        total_reward += r1
        env.render()

        if is_done:
            break

    return total_reward, actions


# In[25]:


import time
time.sleep(1)
for i in range(10):
    tr, _ = test()
    print(f"Reward for test run #{i+1}: {tr}")
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Credits: DRLND, https://zhuanlan.zhihu.com/p/108034550

#Define actions
RIGHT = 4
LEFT = 5

## Utils
def preprocess_single_frame(image, bkg_color = np.array([144, 72, 17])):
    """
    Converts an image from RGB channel to B&W channels.
    Also performs downscale to 80x80. Performs normalization.
    @Param:
    1. image: (array_like) input image. shape = (210, 160, 3)
    2. bkg_color: (np.array) standard encoding for brown in RGB with alpha = 0.0
    @Return:
    - img: (array_like) B&W, downscaled, normalized image of shape (80x80)
    """
    img = np.mean(image[34:-16:2,::2]-bkg_color, axis=-1)/255.
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

# Utils
def collect_trajectories(envs, policy, tmax=200, nrand=5):
    """collect trajectories for an environment"""
    # number of parallel instances
    n=len(envs.ps)

    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]

    envs.reset()
    
    # start all parallel agents
    envs.step([1]*n)
    
    # perform nrand random steps
    for _ in range(nrand):
        fr1, re1, _, _ = envs.step(np.random.choice([RIGHT, LEFT],n))
        fr2, re2, _, _ = envs.step([0]*n)#advances game 1 frame by doing nothing.
    
    for t in range(tmax):

        # prepare the input
        # preprocess_batch properly converts two frames into 
        # shape (n, 2, 80, 80), the proper input for the policy
        # this is required when building CNN with pytorch
        batch_input = preprocess_batch([fr1,fr2])
        
        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        probs = policy(batch_input).squeeze().cpu().detach().numpy()
        
        action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
        probs = np.where(action==RIGHT, probs, 1.0-probs)
        
        
        # advance the game (0=no action)
        # we take one action and skip game forward
        fr1, re1, is_done, _ = envs.step(action)
        fr2, re2, is_done, _ = envs.step([0]*n)

        reward = re1 + re2
        
        # store the result
        state_list.append(batch_input)
        reward_list.append(reward)
        prob_list.append(probs)
        action_list.append(action)
        
        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if is_done.any():
            break


    # return pi_theta, states, actions, rewards, probability
    return prob_list, state_list, action_list, reward_list


#UTILS
def states_to_prob(policy, states):
    """
    Convert states to probability, passing through the policy.
    @Param:
    1. policy: current policy π.
    2. states: states pulled from trajectory.
    @return:
    probabilities of states occurring.
    """
    states = torch.stack(states)
    policy_input = states.view(-1,*states.shape[-3:])
    return policy(policy_input).view(states.shape[:-3])

# UTILS
def clipped_surrogate(policy, old_probs, SAR, discount=0.995, epsilon=0.1, beta=0.01):
    """
    Computes the clipped PPO objective function
    @Param:
    1. policy: current policy
    2. old_probs: probability from the old policy.
    3. SAR: (tuple) trajectory, (States, Actions, Rewards)
    4. discount: discounted return factor.
    5. epsilon: baseline clipped range hyperparameter. range b/w 0.1-0.3
    6. beta: constant regularizer for entropy calculation.
    @Return:
    L_clip = clipped surrogate loss function.
    Establishes a more conservative reward than regular surrogate f(x)
    """
    states, actions, rewards = SAR #extract from tuple
    discounts = discount**np.arange(len(rewards)) # γ^0 + γ^1 + ... + γ^n-1
    rewards = np.asarray(rewards)*discounts[:,np.newaxis] #immediate reward in-place calculation
    
    # convert rewards to future rewards (tackles Credit assignment problem that REINFORCE has)
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

    #perform Z-score based standarization
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1e-10 #prevent from sigma = 0 when normalizing
    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # tensor typecasting
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)#sets conservative rewards

    # convert states to probabilities
    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs) #calculate theta, theta_prime
    
    # ratio for clipping
    ratio = new_probs/old_probs #importance sampling ratio

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

    # entropy: regularization term
    # this steers new_policy towards 0.5
    # add in 1e-10 to avoid log(0) which gives nan
    log_old_probs_theta = torch.log(old_probs+1e-10)
    log_log_probs_theta_prime = torch.log(1.0-old_probs+1e-10)
    entropy = -(new_probs*log_old_probs_theta + (1.0-new_probs)*log_log_probs_theta_prime)

    
    # this returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T
    # averaged over time-step and number of trajectories
    # this is desirable because we have normalized our rewards
    return torch.mean(clipped_surrogate + beta*entropy)


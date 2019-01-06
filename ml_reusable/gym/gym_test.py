import gym

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# TODO
# nothings done but abandoned
def print_reward_heatmap_through_time():
    # Wanted to visualize allreward trajectories through time
    # 2d spectrogram. 
    reward_trajectories = []
    for ep in epoch_stats:
        reward_trajectory = []
        for step in ep:
            reward_trajectory.append(step[1])  # append reward
        reward_trajectories.append(reward_trajectory)
    

    plt.figure()
    plt.hist2d([np.arange(len(tr)) for tr in reward_trajectories], reward_trajectories)
    # plt.hist(epoch_rewards)
    # for trajectory in epoch_rewards:
    #     plt.plot(trajectory)
    plt.show()

    lens = [len(ep) for ep in epoch_stats]


def collect_random_episodes(env, episodes=100, verbose=False, render=False):
    Step = namedtuple('Step', ['obs', 'next_obs', 'action', 'reward', 'done'])

    epoch_rewards = [] 
    epoch_stats = [] 
    i = 0
    for i in tqdm(range(episodes), f'episode {i}/{episodes}'):
            episode_stats = []

            # Reset the environment
            next_obs = env.reset()

            R = 0
            while True:
                obs = next_obs  # Observation that yields action
                action = env.action_space.sample()
                next_obs, reward, done, _ = env.step(action)
                episode_stats.append(Step(obs, next_obs, action, reward, done))
                R += reward

                if verbose:
                    print('Obs: ', next_obs)
                    print('Reward : ', reward)
                    print('Done: ', done)

                if render:
                    env.render()

                if done:
                    epoch_rewards.append(R)
                    epoch_stats.append(episode_stats)
                    break

    return epoch_stats, epoch_rewards


class Model(nn.Module):
    def __init__(self, input_size, action_size, hidden_size, bidirectional=True):
        super(Model, self).__init__()
        self.input_size = input_size
        self.action_size = action_size

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first, bidirectional)
        self.action_out = nn.Linear(hidden_size, action_size)
        self.value_out = nn.Linear(hidden_size, 1)

    def reset(self):



    def forward(self, x):
        z = F.relu(self.rnn(x))
        action = self.action_out(z)
        value = self.value_out(z)
        return x



if __name__ == "__main__":

    # Environment. Defining the task
    env = gym.make('Copy-v0')

    env = gym.make('CartPole-v0')

    # Gather a sense of random reward trajectories
    epoch_stats, epoch_rewards = collect_random_episodes(env, episodes=10000, verbose=False)

    print('Ten first episodes')
    print(epoch_stats[:3])  # namedtuple

    reward_hist = np.histogram(epoch_rewards)
    print(reward_hist)  # namedtuple

    r = plt.hist(epoch_rewards)
    plt.title('Epoch Rewards')
    plt.show()


    # Model Paramaters
    input_size = env.observation_space.shape if env.observation_space.shape is not () else 1
    input_size = input_size[0] if len(input_size) is 1 else input_size

    hidden_size = input_size
    num_layers = 1
    batch_first = True
    bidirectional = True

    model = nn.LSTM(input_size, hidden_size,
            num_layers, batch_first, bidirectional)



env.reset()



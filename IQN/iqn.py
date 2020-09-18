import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gym
import math
from collections import deque
from torch.utils.tensorboard import SummaryWriter


class replay_buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)
        self.memory.append([observation, action, reward, next_observation, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observation, action, reward, next_observation, done = zip(* batch)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done

    def __len__(self):
        return len(self.memory)


class net(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(net, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.feature_fc1 = nn.Linear(self.observation_dim, 128)
        self.feature_fc2 = nn.Linear(128, 128)

        self.phi = nn.Linear(1, 128, bias=False)
        self.phi_bias = nn.Parameter(torch.zeros(128), requires_grad=True)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, self.action_dim)

    def forward(self, observation, sample_num):
        x = F.relu(self.feature_fc1(observation))
        x = F.relu(self.feature_fc2(x))

        tau = torch.rand(sample_num, 1)
        # * tau is the quantile vector
        quants = torch.arange(0, sample_num, 1.0)

        cos_trans = torch.cos(quants * tau * np.pi).unsqueeze(2)
        # * cos_trans: [sample_num, sample_num, 1]
        rand_feat = F.relu(self.phi(cos_trans).mean(1) + self.phi_bias.unsqueeze(0)).unsqueeze(0)
        # * rand_feat: [1, sample_num, 128]
        x = x.unsqueeze(1)
        # * x: [batch_size, 1, 128]
        x = x * rand_feat
        # * x: [batch_size, sample_num, 128]
        x = F.relu(self.fc1(x))
        # * x: [batch_size, sample_num, 128]
        value = self.fc2(x).transpose(1, 2)
        # * value: [batch_size, action_dim, sample_num]
        return value, tau

    def act(self, observation, k_sample, epsilon):
        if random.random() > epsilon:
            value, tau = self.forward(observation, k_sample)
            # * beta is set to be an identity mapping here so calculate the expectation
            action = value.mean(2).max(1)[1].detach().item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action

class iqn(object):
    def __init__(self, env, capacity, episode, exploration, k_sample, k, n, n_prime, gamma, batch_size, learning_rate, epsilon_init, decay, epsilon_min, update_freq, render, log):
        # * n, n_prime, k_sample is N, N', K in the paper
        self.env = env
        self.capacity = capacity
        self.episode = episode
        self.exploration = exploration
        self.k = k
        self.k_sample = k_sample
        self.n = n
        self.n_prime = n_prime
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon_init = epsilon_init
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.update_freq = update_freq
        self.render = render
        self.log = log

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.net = net(self.observation_dim, self.action_dim)
        self.target_net = net(self.observation_dim, self.action_dim)
        self.target_net.load_state_dict(self.net.state_dict())
        self.buffer = replay_buffer(self.capacity)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.epsilon = lambda x: self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(-1. * x / self.decay)
        self.writer = SummaryWriter('run/iqn')
        self.count = 0
        self.weight_reward = None

    def computer_loss(self, tau, value, target_value):
        # * get the quantile huber loss
        u = target_value.unsqueeze(1) - value.unsqueeze(-1)
        huber_loss = 0.5 * u.abs().clamp(min=0., max=self.k).pow(2)
        huber_loss = huber_loss + self.k * (u.abs() - u.abs().clamp(min=0., max=self.k) - 0.5 * self.k)
        quantile_loss = (tau.unsqueeze(0) - (u < 0).float()).abs() * huber_loss
        loss = quantile_loss.mean()
        return loss

    def train(self):
        observations, actions, rewards, next_observations, dones = self.buffer.sample(self.batch_size)

        observations = torch.FloatTensor(observations)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).expand(self.batch_size, self.n_prime)
        next_observations = torch.FloatTensor(next_observations)
        dones = torch.FloatTensor(dones).unsqueeze(1).expand(self.batch_size, self.n_prime)

        dist, tau = self.net.forward(observations, self.n)
        value = dist.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(self.batch_size, 1, self.n)).squeeze()

        target_dist, target_tau = self.target_net.forward(next_observations, self.n_prime)
        target_actions = target_dist.sum(2).max(1)[1].detach()
        target_value = target_dist.gather(1, target_actions.unsqueeze(1).unsqueeze(2).expand(self.batch_size, 1, self.n_prime)).squeeze()
        target_value = rewards + self.gamma * target_value * (1. - dones)
        target_value = target_value.detach()

        loss = self.computer_loss(tau, value, target_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def run(self):
        for i in range(self.episode):
            obs = self.env.reset()
            if self.render:
                self.env.render()
            total_reward = 0
            while True:
                epsilon = self.epsilon(self.count)
                action = self.net.act(torch.FloatTensor(np.expand_dims(obs, 0)), self.k_sample, epsilon)
                next_obs, reward, done, info = self.env.step(action)
                self.count += 1
                total_reward += reward
                if self.render:
                    self.env.render()
                self.buffer.store(obs, action, reward, next_obs, done)
                obs = next_obs

                if self.count > self.exploration:
                    self.train()

                if done:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward
                    print('episode: {}  reward: {}  weight_reward: {:.2f}  epsilon: {:.2f}'.format(i + 1, total_reward, self.weight_reward, epsilon))
                    break


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    test = iqn(env=env,
               capacity=10000,
               episode=100000,
               exploration=1000,
               k_sample=32,
               k=1.,
               n=8,
               n_prime=8,
               gamma=0.99,
               batch_size=32,
               learning_rate=1e-3,
               epsilon_init=1.,
               decay=5000,
               epsilon_min=0.01,
               update_freq=200,
               render=False,
               log=False)
    test.run()
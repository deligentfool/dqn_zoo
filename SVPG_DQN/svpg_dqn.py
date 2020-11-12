import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import gym
import matplotlib.pyplot as plt
import math


def set_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def _square_dist(x):
    # * x: [num_sample, feature_dim]
    xxT = torch.mm(x, x.t())
    # * xxT: [num_sample, num_sample]
    xTx = xxT.diag()
    # * xTx: [num_sample]
    return xTx + xTx.unsqueeze(1) - 2. * xxT


def _Kxx_dxKxx(x, num_agent):
    square_dist = _square_dist(x)
    # * bandwidth = 2 * (med ^ 2)
    bandwidth = 2 * square_dist.median() / math.log(num_agent)
    Kxx = torch.exp(-1. / bandwidth * square_dist)

    dxKxx = 2 * (Kxx.sum(1).diag() - Kxx).matmul(x) / bandwidth

    return Kxx, dxKxx


def calc_returns(rewards, gamma):
    R = 0
    returns = []
    for r in reversed(rewards):
        R += gamma * r
        returns.insert(0, R)
    return returns


def _check_param_device(param, old_param_device):
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:
            warn = (param.get_device() != old_param_device)
        else:
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Parameters on different types!')
    return old_param_device


def parameters_to_vector(parameters, grad=False, both=False):
    param_device = None
    if not both:
        vec = []
        if not grad:
            for param in parameters:
                param_device = _check_param_device(param, param_device)
                vec.append(param.view(-1))

        else:
            for param in parameters:
                param_device = _check_param_device(param, param_device)
                vec.append(param.grad.detach().view(-1))
        return torch.cat(vec)

    else:
        param_vec = []
        grad_vec = []
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            param_vec.append(param.view(-1))
            grad_vec.append(param.grad.detach().view(-1))
        return torch.cat(param_vec), torch.cat(grad_vec)


def vector_to_parameters(vector, parameters, grad=True):
    param_device = None
    pointer = 0

    if grad:
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.grad.data = vector[pointer: pointer + num_param].view(param.size())
            pointer += num_param
    else:
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.data = vector[pointer: pointer + num_param].view(param.size())
            pointer += num_param


def uniform_kernel(X, Y, h):
    particle_num = Y.size(-1)
    distance = (X.unsqueeze(-1) - Y.unsqueeze(-2)).abs()
    kernel_distance = (distance / h) <= 0.5
    prob = kernel_distance.sum(-1) / (particle_num * h)
    return prob


def gaussian_kernel(X, Y, h=None):
    particle_num = Y.size(-1)
    distance = X.unsqueeze(-1) - Y.unsqueeze(-2)
    if h is None:
        h = 2 * distance.pow(2).median().detach()
    kernel_distance = (-0.5 * (distance / h).pow(2)).exp() / np.sqrt(2 * np.pi)
    prob = kernel_distance.sum(-1) / (particle_num * h)
    return prob


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
        observations, actions, rewards, next_observations, dones = zip(* batch)
        return np.concatenate(observations, 0), actions, rewards, np.concatenate(next_observations, 0), dones

    def __len__(self):
        return len(self.memory)


class mmd_ddqn(nn.Module):
    def __init__(self, observation_dim, action_dim, particle_num):
        super(mmd_ddqn, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.particle_num = particle_num
        self.fc1 = nn.Linear(self.observation_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.action_dim * self.particle_num)

    def forward(self, observation):
        x = self.fc1(observation)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x.view(x.size(0), self.action_dim, self.particle_num)

    def act(self, observation, epsilon):
        if random.random() > epsilon:
            q_value = self.forward(observation).mean(-1)
            action = q_value.max(1)[1].data[0].item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action



class svpg_dqn(object):
    def __init__(self, envs, gamma, learning_rate, episode, render, temperature, capacity, batch_size, h, exploration=100, decay=1000, epsilon_min=0.05, epsilon_init=1., particle_num=200, update_freq=300, max_episode_length=200):
        self.envs = envs
        self.num_agent = len(self.envs)
        self.observation_dim = self.envs[0].observation_space.shape[0]
        self.action_dim = self.envs[0].action_space.n
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.episode = episode
        self.render = render
        self.temperature = temperature
        self.particle_num = particle_num
        self.capacity = capacity
        self.exploration = exploration
        self.buffers = [replay_buffer(self.capacity) for _ in range(self.num_agent)]
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.h = h
        self.eps = np.finfo(np.float32).eps.item()
        self.policies = [mmd_ddqn(self.observation_dim, self.action_dim, self.particle_num) for _ in range(self.num_agent)]
        self.target_policies = [mmd_ddqn(self.observation_dim, self.action_dim, self.particle_num) for _ in range(self.num_agent)]
        [target_policy.load_state_dict(policy.state_dict()) for target_policy, policy in zip(self.target_policies, self.policies)]
        self.optimizers = [torch.optim.Adam(self.policies[i].parameters(), lr=self.learning_rate) for i in range(self.num_agent)]
        self.total_returns = []
        self.weight_reward = None
        self.max_episode_length = max_episode_length
        self.epsilon_func = lambda e: epsilon_init - (epsilon_init - epsilon_min) / decay * min(e, decay)


    def train(self):
        policy_grads = []
        parameters = []

        for i in range(self.num_agent):
            observation, action, reward, next_observation, done = self.buffers[i].sample(self.batch_size)

            observation = torch.FloatTensor(observation)
            action = torch.LongTensor(action)
            reward = torch.FloatTensor(reward)
            next_observation = torch.FloatTensor(next_observation)
            done = torch.FloatTensor(done)

            q_particle_values = self.policies[i].forward(observation)
            next_q_particle_values = self.target_policies[i].forward(next_observation)
            argmax_actions = self.policies[i].forward(next_observation).mean(-1).max(1)[1].detach()
            next_q_particle_value = next_q_particle_values.gather(1, argmax_actions.unsqueeze(1).unsqueeze(-1).repeat([1, 1, next_q_particle_values.size(-1)])).squeeze(1)
            q_particle_value = q_particle_values.gather(1, action.unsqueeze(1).unsqueeze(-1).repeat([1, 1, q_particle_values.size(-1)])).squeeze(1)
            expected_q_particle_value = (reward.unsqueeze(-1) + self.gamma * (1 - done.unsqueeze(-1)) * next_q_particle_value).detach()

            agent_policy_grad = (gaussian_kernel(q_particle_value, expected_q_particle_value, self.h)).log()
            #for log_prob, r in zip(self.policies[i].log_probs, returns):
            #    agent_policy_grad.append(log_prob * r)

            self.optimizers[i].zero_grad()

            policy_grad = agent_policy_grad.sum()
            policy_grad.backward()

            param_vector, grad_vector = parameters_to_vector(self.policies[i].parameters(), both=True)
            policy_grads.append(grad_vector.unsqueeze(0))
            parameters.append(param_vector.unsqueeze(0))

        parameters = torch.cat(parameters)
        Kxx, dxKxx = _Kxx_dxKxx(parameters, self.num_agent)
        policy_grads = 1. / self.temperature * torch.cat(policy_grads)
        grad_logp = torch.mm(Kxx, policy_grads)
        grad_theta = - (grad_logp.detach() + dxKxx) / self.num_agent

        for i in range(self.num_agent):
            vector_to_parameters(grad_theta[i], self.policies[i].parameters(), grad=True)
            self.optimizers[i].step()


    def run(self):
        for i_episode in range(self.episode):
            max_reward = -np.inf
            for i, env in enumerate(self.envs):
                obs = env.reset()
                total_reward = 0
                count = 0
                if self.render:
                    env.render()
                while count < self.max_episode_length:
                    action = self.policies[i].act(torch.FloatTensor(np.expand_dims(obs, 0)), self.epsilon_func(i_episode))
                    next_obs, reward, done, info = env.step(action)
                    if count == self.max_episode_length - 1:
                        done = True
                    self.buffers[i].store(obs, action, reward, next_obs, done)
                    total_reward += reward
                    count += 1
                    if self.render:
                        env.render()
                    obs = next_obs
                    if done:
                        break
                if max_reward < total_reward:
                    max_reward = total_reward
            if self.weight_reward is None:
                self.weight_reward = max_reward
            else:
                self.weight_reward = 0.99 * self.weight_reward + 0.01 * max_reward
            print('episode: {}\t max_reward: {:.1f}\t weight_reward: {:.2f}'.format(i_episode + 1, max_reward, self.weight_reward))
            if i_episode >= self.exploration - 1:
                self.train()
                if i_episode % self.update_freq == 0:
                    [target_policy.load_state_dict(policy.state_dict()) for target_policy, policy in zip(self.target_policies, self.policies)]

if __name__ == '__main__':
    num_agent = 8
    seed = 2020
    set_seed(seed)
    envs = [gym.make('CartPole-v0') for _ in range(num_agent)]
    envs = [env.unwrapped for env in envs]
    [env.seed(seed) for env in envs]
    test = svpg_dqn(
        envs,
        gamma=0.99,
        learning_rate=1e-3,
        episode=100000,
        render=False,
        temperature=5.0,
        capacity=50000,
        batch_size=64,
        h=1
    )
    test.run()
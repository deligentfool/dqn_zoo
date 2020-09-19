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


class fqf_net(nn.Module):
    def __init__(self, observation_dim, action_dim, quant_num, cosine_num):
        super(fqf_net, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.quant_num = quant_num
        self.cosine_num = cosine_num

        self.feature_layer = nn.Sequential(
            nn.Linear(self.observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.cosine_layer = nn.Sequential(
            nn.Linear(self.cosine_num, 128),
            nn.ReLU()
        )

        self.psi_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )

        self.quantile_fraction_layer = nn.Sequential(
            nn.Linear(128, self.quant_num),
            nn.Softmax(dim=-1)
        )

    def calc_state_embedding(self, observation):
        return self.feature_layer(observation)

    def calc_quantile_fraction(self, state_embedding):
        assert not state_embedding.requires_grad
        q = self.quantile_fraction_layer(state_embedding.detach())
        tau_0 = torch.zeros(q.size(0), 1)
        tau = torch.cat([tau_0, q], dim=-1)
        tau = torch.cumsum(tau, dim=-1)
        entropy = torch.distributions.Categorical(probs=q).entropy()
        tau_hat = ((tau[:, :-1] + tau[:, 1:]) / 2.).detach()
        return tau, tau_hat, entropy

    def calc_quantile_value(self, tau, state_embedding):
        assert not tau.requires_grad
        quants = torch.arange(0, self.cosine_num, 1.0).unsqueeze(0).unsqueeze(0)
        cos_trans = torch.cos(quants * tau.unsqueeze(-1).detach() * np.pi)
        # * cos_trans: [batch_size, quant_num, cosine_num]
        rand_feat = self.cosine_layer(cos_trans)
        # * rand_feat: [batch_size, quant_num, 128]
        x = state_embedding.unsqueeze(1)
        # * x: [batch_size, 1, 128]
        x = x * rand_feat
        # * x: [batch_size, quant_num, 128]
        value = self.psi_layer(x).transpose(1, 2)
        # * value: [batch_size, action_dim, quant_num]
        return value

    def act(self, observation, epsilon):
        if random.random() > epsilon:
            state_embedding = self.calc_state_embedding(observation)
            tau, tau_hat, _ = self.calc_quantile_fraction(state_embedding.detach())
            q_value = self.calc_q_value(state_embedding, tau, tau_hat)
            action = q_value.max(1)[1].detach().item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action

    def calc_sa_quantile_value(self, state_embedding, action, tau):
        sa_quantile_value = self.calc_quantile_value(tau.detach(), state_embedding)
        sa_quantile_value = sa_quantile_value.gather(1, action.unsqueeze(-1).unsqueeze(-1).expand(sa_quantile_value.size(0), 1, sa_quantile_value.size(-1))).squeeze(1)
        return sa_quantile_value

    def calc_q_value(self, state_embedding, tau, tau_hat):
        tau_delta = tau[:, 1:] - tau[:, :-1]
        tau_hat_value = self.calc_quantile_value(tau_hat.detach(), state_embedding)
        q_value = (tau_delta.unsqueeze(1) * tau_hat_value).sum(-1).detach()
        return q_value



class fqf(object):
    def __init__(self, env, capacity, episode, exploration, k, gamma, quant_num, cosine_num, batch_size, value_learning_rate, fraction_learning_rate, entropy_weight, epsilon_init, double_q, decay, epsilon_min, update_freq, render, log):
        self.env = env
        self.capacity = capacity
        self.episode = episode
        self.exploration = exploration
        self.k = k
        self.gamma = gamma
        self.batch_size = batch_size
        self.value_learning_rate = value_learning_rate
        self.fraction_learning_rate = fraction_learning_rate
        self.epsilon_init = epsilon_init
        self.decay = decay
        self.quant_num = quant_num
        self.epsilon_min = epsilon_min
        self.entropy_weight = entropy_weight
        self.update_freq = update_freq
        self.render = render
        self.log = log
        self.cosine_num = cosine_num
        self.double_q = double_q

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.net = fqf_net(self.observation_dim, self.action_dim, self.quant_num, self.cosine_num)
        self.target_net = fqf_net(self.observation_dim, self.action_dim, self.quant_num, self.cosine_num)
        self.target_net.load_state_dict(self.net.state_dict())
        self.buffer = replay_buffer(self.capacity)
        self.quantile_value_param = list(self.net.feature_layer.parameters()) + list(self.net.cosine_layer.parameters()) + list(self.net.psi_layer.parameters())
        self.quantile_fraction_param = list(self.net.quantile_fraction_layer.parameters())
        self.quantile_value_optimizer = torch.optim.Adam(self.quantile_value_param, lr=self.value_learning_rate)
        self.quantile_fraction_optimizer = torch.optim.RMSprop(self.quantile_fraction_param, lr=self.fraction_learning_rate)
        self.epsilon = lambda x: self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(-1. * x / self.decay)
        self.writer = SummaryWriter('run/fqf')
        self.count = 0
        self.weight_reward = None

    def calc_quantile_value_loss(self, tau, value, target_value):
        # * calculate quantile value loss
        # * get the quantile huber loss
        assert not tau.requires_grad
        u = target_value.unsqueeze(1) - value.unsqueeze(-1)
        huber_loss = 0.5 * u.abs().clamp(min=0., max=self.k).pow(2)
        huber_loss = huber_loss + self.k * (u.abs() - u.abs().clamp(min=0., max=self.k) - 0.5 * self.k)
        quantile_loss = (tau.unsqueeze(-1) - (u < 0).float()).abs() * huber_loss
        loss = quantile_loss.mean()
        return loss

    def calc_quantile_fraction_loss(self, observations, actions, tau, tau_hat):
        # * calculate quantile fraction loss
        assert not tau_hat.requires_grad
        sa_quantile_hat = self.net.calc_sa_quantile_value(observations, actions, tau_hat).detach()
        sa_quantile = self.net.calc_sa_quantile_value(observations, actions, tau[:, 1:-1]).detach()
        gradient_tau = 2 * sa_quantile - sa_quantile_hat[:, :-1] - sa_quantile_hat[:, 1:]
        return (gradient_tau.detach() * tau[:, 1: -1]).sum(1).mean()


    def train(self):
        observations, actions, rewards, next_observations, dones = self.buffer.sample(self.batch_size)

        observations = torch.FloatTensor(observations)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_observations = torch.FloatTensor(next_observations)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        state_embedding = self.net.calc_state_embedding(observations)
        tau, tau_hat, entropy = self.net.calc_quantile_fraction(state_embedding.detach())
        dist = self.net.calc_quantile_value(tau_hat.detach(), state_embedding)
        value = dist.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(self.batch_size, 1, dist.size(2))).squeeze()

        if not self.double_q:
            next_state_embedding = self.target_net.calc_state_embedding(next_observations)
            # * have to use the eval_net's quantile fraction network
            next_tau, next_tau_hat, _ = self.net.calc_quantile_fraction(next_state_embedding.detach())
            target_actions = self.target_net.calc_q_value(next_state_embedding, next_tau, next_tau_hat).max(1)[1].detach()
        else:
            next_state_embedding = self.net.calc_state_embedding(next_observations)
            next_tau, next_tau_hat, _ = self.net.calc_quantile_fraction(next_state_embedding.detach())
            target_actions = self.net.calc_q_value(next_state_embedding, next_tau, next_tau_hat).max(1)[1].detach()
        next_state_embedding = self.target_net.calc_state_embedding(next_observations)
        target_dist = self.target_net.calc_quantile_value(tau_hat.detach(), next_state_embedding)
        target_value = target_dist.gather(1, target_actions.unsqueeze(1).unsqueeze(2).expand(self.batch_size, 1, target_dist.size(2))).squeeze()
        target_value = rewards + self.gamma * target_value * (1. - dones)
        target_value = target_value.detach()

        qauntile_value_loss = self.calc_quantile_value_loss(tau_hat.detach(), value, target_value)
        quantile_fraction_loss = self.calc_quantile_fraction_loss(state_embedding, actions, tau, tau_hat)
        entropy_loss = - (self.entropy_weight * entropy).mean()

        self.quantile_fraction_optimizer.zero_grad()
        quantile_fraction_loss.backward(retain_graph=True)
        self.quantile_fraction_optimizer.step()

        self.quantile_value_optimizer.zero_grad()
        qauntile_value_loss.backward()
        self.quantile_value_optimizer.step()

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
                action = self.net.act(torch.FloatTensor(np.expand_dims(obs, 0)), epsilon)
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
    torch.autograd.set_detect_anomaly(True)
    test = fqf(env=env,
               capacity=10000,
               episode=100000,
               exploration=1000,
               k=1.,
               gamma=0.99,
               batch_size=32,
               quant_num=32,
               cosine_num=64,
               value_learning_rate=1e-3,
               fraction_learning_rate=1e-9,
               entropy_weight=0,
               double_q=True,
               epsilon_init=1,
               decay=5000,
               epsilon_min=0.01,
               update_freq=200,
               render=False,
               log=False)
    test.run()
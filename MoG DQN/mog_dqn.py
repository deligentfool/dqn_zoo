import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque


def set_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


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


class MDN(nn.Module):
    def __init__(self, in_features, out_features, num_gaussians, hidden_dim):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.hidden_dim = hidden_dim
        self.nn_layer = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_features * self.num_gaussians * 3)
        )

    def forward(self, minibatch):
        out = self.nn_layer(minibatch)
        out = out.view(-1, 3, self.out_features, self.num_gaussians)
        pi = out[:, 0]
        pi = F.softmax(pi, dim=-1)
        sigma = torch.exp(out[:, 1]) + 0.01
        mu = out[:, 2]
        return pi, sigma, mu

    def act(self, input, epsilon):
        if random.random() > epsilon:
            pi, sigma, mu = self.forward(input)
            q_value = (pi * mu).sum(-1)
            action = q_value.max(1)[1].data[0].item()
        else:
            action = random.choice(list(range(self.out_features)))
        return action


def gaussian_func(mu, sigma_2, x):
    return (1. / torch.sqrt(2 * np.pi * sigma_2)) * (- (x - mu).pow(2) / (2 * sigma_2)).exp()


def train(buffer, target_model, eval_model, gamma, optimizer, batch_size, count, update_freq):
    observation, action, reward, next_observation, done = buffer.sample(batch_size)

    observation = torch.FloatTensor(observation)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    next_observation = torch.FloatTensor(next_observation)
    done = torch.FloatTensor(done)

    pi, sigma, mu = eval_model.forward(observation)
    num_gaussians = mu.size(-1)
    sigma = sigma.gather(1, action.unsqueeze(1).unsqueeze(-1).repeat([1, 1, num_gaussians])).squeeze()
    mu = mu.gather(1, action.unsqueeze(1).unsqueeze(-1).repeat([1, 1, num_gaussians])).squeeze()
    pi = pi.gather(1, action.unsqueeze(1).unsqueeze(-1).repeat([1, 1, num_gaussians])).squeeze()
    next_pi, next_sigma, next_mu = target_model.forward(next_observation)
    argmax_pi, argmax_sigma, argmax_mu = eval_model.forward(next_observation)
    argmax_actions = (argmax_pi * argmax_mu).sum(-1).max(1)[1].detach()

    next_sigma = next_sigma.gather(1, argmax_actions.unsqueeze(1).unsqueeze(-1).repeat([1, 1, num_gaussians])).squeeze()
    next_mu = next_mu.gather(1, argmax_actions.unsqueeze(1).unsqueeze(-1).repeat([1, 1, num_gaussians])).squeeze()
    next_pi = next_pi.gather(1, argmax_actions.unsqueeze(1).unsqueeze(-1).repeat([1, 1, num_gaussians])).squeeze()
    next_mu = gamma * next_mu * (1. - done.unsqueeze(-1)) + reward.unsqueeze(-1)
    next_sigma = gamma * gamma * next_sigma
    next_mu, next_sigma, next_pi = next_mu.detach(), next_sigma.detach(), next_pi.detach()

    mu_i = mu.unsqueeze(-1).repeat([1, 1, num_gaussians])
    mu_j = mu.unsqueeze(-2).repeat([1, num_gaussians, 1])
    sigma_i = sigma.unsqueeze(-1).repeat([1, 1, num_gaussians])
    sigma_j = sigma.unsqueeze(-2).repeat([1, num_gaussians, 1])
    pi_i = pi.unsqueeze(-1).repeat([1, 1, num_gaussians])
    pi_j = pi.unsqueeze(-2).repeat([1, num_gaussians, 1])

    next_mu_i = next_mu.unsqueeze(-1).repeat([1, 1, num_gaussians])
    next_mu_j = next_mu.unsqueeze(-2).repeat([1, num_gaussians, 1])
    next_sigma_i = next_sigma.unsqueeze(-1).repeat([1, 1, num_gaussians])
    next_sigma_j = next_sigma.unsqueeze(-2).repeat([1, num_gaussians, 1])
    next_pi_i = next_pi.unsqueeze(-1).repeat([1, 1, num_gaussians])
    next_pi_j = next_pi.unsqueeze(-2).repeat([1, num_gaussians, 1])

    first_item = pi_i * pi_j * gaussian_func(mu_j, sigma_i + sigma_j, mu_i)
    second_item = next_pi_i * next_pi_j * gaussian_func(next_mu_j, next_sigma_i + next_sigma_j, next_mu_i)
    third_item = pi_i * next_pi_j * gaussian_func(next_mu_j, sigma_i + next_sigma_j, mu_i)

    jtd_loss = first_item.sum(-1).sum(-1) + second_item.sum(-1).sum(-1) - 2 * third_item.sum(-1).sum(-1)
    jtd_loss = jtd_loss.mean()

    optimizer.zero_grad()
    jtd_loss.backward()
    optimizer.step()
    if count % update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())


if __name__ == '__main__':
    gamma = 0.99
    learning_rate = 3e-4
    batch_size = 64
    update_freq = 500
    capacity = 50000
    exploration = 200
    epsilon_init = 0.5
    epsilon_min = 0.02
    decay = 0.995
    episode = 1000000
    render = False
    num_gaussians = 11
    hidden_dim = 256

    seed = 2022
    set_seed(seed)
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    env.seed(seed)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    target_net = MDN(observation_dim, action_dim, num_gaussians, hidden_dim)
    eval_net = MDN(observation_dim, action_dim, num_gaussians, hidden_dim)
    target_net.load_state_dict(eval_net.state_dict())
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)
    buffer = replay_buffer(capacity)
    epsilon = epsilon_init
    count = 0

    weight_reward = None
    for i in range(episode):
        obs = env.reset()
        if epsilon > epsilon_min:
            epsilon = epsilon * decay
        reward_total = 0
        if render:
            env.render()
        while True:
            action = eval_net.act(torch.FloatTensor(np.expand_dims(obs, 0)), epsilon)
            count += 1
            next_obs, reward, done, info = env.step(action)
            buffer.store(obs, action, reward, next_obs, done)
            reward_total += reward
            obs = next_obs
            if render:
                env.render()
            if i > exploration:
                train(buffer, target_net, eval_net, gamma, optimizer, batch_size, count, update_freq)

            if done:
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                print('episode: {}\tepsilon: {:.2f}\treward: {:.2f}\tweight_reward: {:.3f}'.format(i+1, epsilon, reward_total, weight_reward))
                break



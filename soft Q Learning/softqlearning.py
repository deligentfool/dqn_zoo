import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import gym


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


class soft_q_net(nn.Module):
    def __init__(self, observation_dim, action_dim, alpha):
        super(soft_q_net, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.fc1 = nn.Linear(self.observation_dim, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, self.action_dim)

    def forward(self, observation):
        x = self.fc1(observation)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def act(self, observation):
        with torch.no_grad():
            q_value = self.forward(observation)
            v = self.getV(q_value)
            pi_maxent = torch.exp((q_value - v) / self.alpha)
            pi_maxent = pi_maxent / pi_maxent.sum(dim=-1, keepdim=True)
            dist = torch.distributions.Categorical(pi_maxent)
            action = dist.sample().item()
        return action

    def getV(self, q_value):
        v = self.alpha * torch.log((1 / self.alpha * q_value).exp().sum(dim=-1, keepdim=True))
        return v


def train(buffer, target_model, eval_model, gamma, optimizer, batch_size, loss_fn, count, update_freq):
    observation, action, reward, next_observation, done = buffer.sample(batch_size)

    observation = torch.FloatTensor(observation)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    next_observation = torch.FloatTensor(next_observation)
    done = torch.FloatTensor(done)

    q_values = eval_model.forward(observation)
    next_q_values = target_model.forward(next_observation)
    next_v_values = target_model.getV(next_q_values)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * (1 - done) * next_v_values.squeeze(-1)

    #loss = loss_fn(q_value, expected_q_value.detach())
    loss = (expected_q_value.detach() - q_value).pow(2)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if count % update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())


if __name__ == '__main__':
    gamma = 0.99
    learning_rate = 1e-4
    batch_size = 32
    update_freq = 200
    capacity = 50000
    render = False
    episode = 100000
    alpha = 4

    env = gym.make('CartPole-v0')
    env = env.unwrapped
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    target_net = soft_q_net(observation_dim, action_dim, alpha)
    eval_net = soft_q_net(observation_dim, action_dim, alpha)
    eval_net.load_state_dict(target_net.state_dict())
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)
    buffer = replay_buffer(capacity)
    loss_fn = nn.MSELoss()
    count = 0

    weight_reward = None
    for i in range(episode):
        obs = env.reset()
        reward_total = 0
        if render:
            env.render()
        while True:
            action = eval_net.act(torch.FloatTensor(np.expand_dims(obs, 0)))
            count += 1
            next_obs, reward, done, info = env.step(action)
            buffer.store(obs, action, reward, next_obs, done)
            reward_total += reward
            obs = next_obs
            if render:
                env.render()
            if len(buffer.memory) > batch_size:
                train(buffer, target_net, eval_net, gamma, optimizer, batch_size, loss_fn, count, update_freq)

            if done:
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                if (i+1) % 10 == 0:
                    print('episode: {}\treward: {}\tweight_reward: {:.3f}'.format(i+1, reward_total, weight_reward))
                break


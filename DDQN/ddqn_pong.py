import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
sys.path.append('.')
from collections import deque
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math


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


class cnn_ddqn(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(cnn_ddqn, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.conv1 = nn.Conv2d(self.observation_dim[0], 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.fc2 = nn.Linear(512, self.action_dim)


    def feature_size(self):
        tmp = torch.zeros(1, * self.observation_dim)
        return self.conv3(self.conv2(self.conv1(tmp))).view(1, -1).size(1)


    def forward(self, observation):
        x = self.conv1(observation)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def act(self, observation, epsilon):
        if random.random() > epsilon:
            q_value = self.forward(observation)
            action = q_value.max(1)[1].data[0].item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action


def train(buffer, target_model, eval_model, gamma, optimizer, batch_size, loss_fn, count, soft_update_freq):
    observation, action, reward, next_observation, done = buffer.sample(batch_size)

    observation = torch.FloatTensor(observation)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    next_observation = torch.FloatTensor(next_observation)
    done = torch.FloatTensor(done)

    q_values = eval_model.forward(observation)
    next_q_values = target_model.forward(next_observation)
    argmax_actions = eval_model.forward(next_observation).max(1)[1].detach()
    next_q_value = next_q_values.gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    expected_q_value = (reward + gamma * (1 - done) * next_q_value).detach()

    loss = loss_fn(q_value, expected_q_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if count % soft_update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())


if __name__ == '__main__':
    epsilon_init = 1.0
    epsilon_min = 0.01
    decay = 30000
    epsilon_by_count = lambda idx: epsilon_min + (epsilon_init - epsilon_min) * math.exp(-1. * idx / decay)
    capacity = 100000
    exploration = 10000
    batch_size = 32
    gamma = 0.99
    learning_rate = 1e-4
    soft_update_freq = 1000
    episode = 1000000
    render = True

    env_id = "PongNoFrameskip-v4"
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    action_dim = env.action_space.n
    observation_dim = env.observation_space.shape
    action_dim = env.action_space.n
    target_net = cnn_ddqn(observation_dim, action_dim)
    eval_net = cnn_ddqn(observation_dim, action_dim)
    eval_net.load_state_dict(target_net.state_dict())
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)
    buffer = replay_buffer(capacity)
    loss_fn = nn.MSELoss()
    epsilon = epsilon_init
    count = 0

    weight_reward = None
    for i in range(episode):
        obs = env.reset()
        reward_total = 0
        if render:
            env.render()
        while True:
            epsilon = epsilon_by_count(count)
            action = eval_net.act(torch.FloatTensor(np.expand_dims(obs, 0)), epsilon)
            count += 1
            next_obs, reward, done, info = env.step(action)
            buffer.store(obs, action, reward, next_obs, done)
            reward_total += reward
            obs = next_obs
            if render:
                env.render()
            if len(buffer) > exploration:
                train(buffer, target_net, eval_net, gamma, optimizer, batch_size, loss_fn, count, soft_update_freq)

            if done:
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.9 * weight_reward + 0.1 * reward_total
                print('episode: {}  epsilon: {:.2f}  reward: {}  weight_reward: {:.3f}'.format(i+1, epsilon, reward_total, weight_reward))
                break


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import sys
sys.path.append('.')
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math

class cnn_dqn(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(cnn_dqn, self).__init__()
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

    def get_action(self, observation, epsilon):
        if random.random() > epsilon:
            q_value = self.forward(observation)
            action = q_value.max(1)[1].data[0].item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action


class replay_buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def store(self, observation, action, reward, next_observation, done, ):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)
        self.memory.append([observation, action, reward, next_observation, done])

    def sample(self, size):
        batch = random.sample(self.memory, size)
        observation, action, reward, next_observation, done = zip(* batch)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done

    def __len__(self):
        return len(self.memory)


def training(buffer, batch_size, model, optimizer, gamma, loss_fn):
    observation, action, reward, next_observation, done = buffer.sample(batch_size)

    observation = torch.FloatTensor(observation)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    next_observation = torch.FloatTensor(next_observation)
    done = torch.FloatTensor(done)

    q_values = model.forward(observation)
    next_q_values = model.forward(next_observation)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0].detach()
    expected_q_value = reward + next_q_value * (1 - done) * gamma

    loss = loss_fn(q_value, expected_q_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



if __name__ == '__main__':
    epsilon_init = 1.0
    epsilon_min = 0.01
    decay = 30000
    epsilon_by_count = lambda idx: epsilon_min + (epsilon_init - epsilon_min) * math.exp(-1. * idx / decay)
    capacity = 100000
    exploration = 10000
    batch_size = 32
    episode = 1000000
    render = True
    learning_rate = 1e-5
    gamma = 0.99
    loss_fn = nn.MSELoss()

    env_id = "PongNoFrameskip-v4"
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    action_dim = env.action_space.n
    observation_dim = env.observation_space.shape

    model = cnn_dqn(observation_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    buffer = replay_buffer(capacity)
    weight_reward = None
    count = 0

    for i in range(episode):
        obs = env.reset()
        reward_total = 0
        if render:
            env.render()
        while True:
            epsilon = epsilon_by_count(count)
            action = model.get_action(torch.FloatTensor(np.expand_dims(obs, 0)), epsilon)
            next_obs, reward, done, info = env.step(action)
            if render:
                env.render()
            buffer.store(obs, action, reward, next_obs, done)
            reward_total += reward
            obs = next_obs
            count += 1
            if len(buffer) > exploration:
                training(buffer, batch_size, model, optimizer, gamma, loss_fn)
            if done:
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                print('episode: {}  reward: {}  epsilon: {:.5f}  weight_reward: {:.3f}'.format(i+1, reward_total, epsilon, weight_reward))
                break

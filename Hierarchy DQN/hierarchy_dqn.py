import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class stochastic_mdp(object):
    def __init__(self):
        self.flag = False
        self.current_state = 2
        self.action_dim = 2
        self.state_dim = 6
        self.right_prob = 0.5

    def reset(self):
        self.flag = False
        self.current_state = 2
        state = np.zeros(self.state_dim)
        state[self.current_state - 1] = 1
        return state

    def step(self, action):
        if action == 1:
            if random.random() < self.right_prob and self.current_state < self.state_dim:
                self.current_state += 1
            else:
                self.current_state -= 1
        elif action == 0:
            self.current_state -= 1
        if self.current_state == self.state_dim:
            self.flag = True

        state = np.zeros(self.state_dim)
        state[self.current_state - 1] = 1

        if self.current_state == 1:
            if self.flag:
                return state, 1., True, {}
            else:
                return state, 1e-2, True, {}
        else:
            return state, 0., False, {}


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


class dqn(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(dqn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, self.output_dim)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.fc2(x)
        return x

    def act(self, input, epsilon):
        if random.random() > epsilon:
            value = self.forward(input)
            action = value.max(1)[1].detach()[0].item()
        else:
            action = random.choice(list(range(self.output_dim)))
        return action


def train(model, buffer, gamma, batch_size, optimizer):
    observation, action, reward, next_observation, done = buffer.sample(batch_size)

    observation = torch.FloatTensor(observation)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    next_observation = torch.FloatTensor(next_observation)
    done = torch.FloatTensor(done)

    values = model.forward(observation)
    value = values.gather(1, action.unsqueeze(1)).squeeze(1)

    next_values = model.forward(next_observation)
    next_value = next_values.max(1)[0].detach()
    expected_value = reward + next_value * (1 - done) * gamma

    loss = (expected_value - value).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def to_onehot(dim, idx):
    state = np.zeros(dim)
    state[idx - 1] = 1
    return state

if __name__ == '__main__':
    learning_rate = 1e-3
    batch_size = 64
    capacity = 10000
    episode = 10000000
    exploration = 300
    gamma = 0.99
    epsilon_init = 0.95
    epsilon_decay = 0.995
    epsilon_min = 0.01

    env = stochastic_mdp()
    action_dim = env.action_dim
    observation_dim = env.state_dim
    epsilon_g = [epsilon_init] * observation_dim
    epsilon = epsilon_init
    controller = dqn(observation_dim * 2, action_dim)
    meta_controller = dqn(observation_dim, observation_dim)
    buffer = replay_buffer(capacity)
    meta_buffer = replay_buffer(capacity)
    optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)
    meta_optimizer = torch.optim.Adam(meta_controller.parameters(), lr=learning_rate)
    weight_reward = None

    for i in range(episode):
        obs = env.reset()
        episode_reward = 0
        while True:
            goal = meta_controller.act(torch.FloatTensor(np.expand_dims(obs, 0)), epsilon)
            onehot_goal = to_onehot(observation_dim, goal)

            meta_obs = obs
            extrinsic_reward = 0

            while True:
                observation_goal = np.concatenate([obs, onehot_goal], 0)
                action = controller.act(torch.FloatTensor(np.expand_dims(observation_goal, 0)), epsilon_g[goal - 1])
                next_obs, reward, done, info = env.step(action)
                extrinsic_reward += reward
                episode_reward += reward
                intrinsic_reward = 1. if goal == np.argmax(next_obs) else 0.
                buffer.store(observation_goal, action, intrinsic_reward, np.concatenate([next_obs, onehot_goal], 0), done)
                obs = next_obs
                if len(buffer) > exploration and len(meta_buffer) > exploration:
                    train(controller, buffer, gamma, batch_size, optimizer)
                    train(meta_controller, meta_buffer, gamma, batch_size, meta_optimizer)
                if goal == np.argmax(next_obs) or done:
                    if epsilon_g[goal - 1] > epsilon_min:
                        epsilon_g[goal - 1] *= epsilon_decay
                    break

            meta_buffer.store(meta_obs, goal, extrinsic_reward, next_obs, done)
            if done:
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay
                if not weight_reward:
                    weight_reward = extrinsic_reward
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * extrinsic_reward
                print('episode: {}  reward: {:.2f}  weight_reward: {:.3f}'.format(i+1, episode_reward, weight_reward))
                break


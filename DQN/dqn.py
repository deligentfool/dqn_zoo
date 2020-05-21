import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import gym

class dqn(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(dqn, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.observation_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.action_dim)

    def forward(self, observation):
        x = self.fc1(observation)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
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

    loss = loss_fn(q_value, expected_q_value.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



if __name__ == '__main__':
    epsilon_init = 0.9
    epsilon_min = 0.01
    decay = 0.995
    capacity = 10000
    exploration = 5000
    batch_size = 64
    episode = 1000000
    render = True
    learning_rate = 1e-3
    gamma = 0.99
    loss_fn = nn.MSELoss()

    env = gym.make('CartPole-v0')
    env = env.unwrapped
    action_dim = env.action_space.n
    observation_dim = env.observation_space.shape[0]

    model = dqn(observation_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    buffer = replay_buffer(capacity)
    epsilon = epsilon_init
    weight_reward = None

    for i in range(episode):
        obs = env.reset()
        if epsilon > epsilon_min:
            epsilon = epsilon * decay
        reward_total = 0
        if render:
            env.render()
        while True:
            action = model.get_action(torch.FloatTensor(np.expand_dims(obs, 0)), epsilon)
            train_flag = False
            next_obs, reward, done, info = env.step(action)
            if render:
                env.render()
            buffer.store(obs, action, reward, next_obs, done)
            reward_total += reward
            obs = next_obs
            if len(buffer) > exploration:
                training(buffer, batch_size, model, optimizer, gamma, loss_fn)
                train_flag = True
            if done:
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                print('episode: {}  reward: {}  epsilon: {:.2f}  train:  {}  weight_reward: {:.3f}'.format(i+1, reward_total, epsilon, train_flag, weight_reward))
                break

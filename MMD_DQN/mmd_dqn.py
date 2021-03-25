import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import gym
import matplotlib.pyplot as plt


def set_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def calc_kernel(X, Y, kernel_num=20):
    first_kernel = (Y.unsqueeze(-1) - Y.unsqueeze(-2)).pow(2)
    second_kernel = (X.unsqueeze(-1) - X.unsqueeze(-2)).pow(2)
    third_kernel = (Y.unsqueeze(-1) - X.unsqueeze(-2)).pow(2)
    bandwidth_list = [2 ** i for i in range(kernel_num)]
    first_items = 0
    second_items = 0
    third_items = 0
    for h in bandwidth_list:
        h = 2 * (h ** 2)
        first_inner_distance = (-first_kernel / h)
        second_inner_distance = (-second_kernel / h)
        intra_distance = (-third_kernel / h)
        first_items += first_inner_distance
        second_items += second_inner_distance
        third_items += intra_distance
    return first_items, second_items, third_items


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
        self.fc1 = nn.Linear(self.observation_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.action_dim * self.particle_num)

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


def train(buffer, eval_model, target_model, gamma, optimizer, batch_size, count, update_freq, device):
    observation, action, reward, next_observation, done = buffer.sample(batch_size)

    observation = torch.FloatTensor(observation).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    next_observation = torch.FloatTensor(next_observation).to(device)
    done = torch.FloatTensor(done).to(device)

    particle_num = eval_model.particle_num
    q_particle_values = eval_model.forward(observation)
    next_q_particle_values = target_model.forward(next_observation)
    argmax_actions = eval_model.forward(next_observation).mean(-1).max(1)[1].detach()
    next_q_particle_value = next_q_particle_values.gather(1, argmax_actions.unsqueeze(1).unsqueeze(-1).repeat([1, 1, next_q_particle_values.size(-1)])).squeeze(1)
    q_particle_value = q_particle_values.gather(1, action.unsqueeze(1).unsqueeze(-1).repeat([1, 1, q_particle_values.size(-1)])).squeeze(1)
    expected_q_particle_value = (reward.unsqueeze(-1) + gamma * (1 - done.unsqueeze(-1)) * next_q_particle_value).detach()

    X = expected_q_particle_value
    Y = q_particle_value
    first_item, second_item, third_item = calc_kernel(X, Y)
    first_item = (first_item.sum(-1).sum(-1) / (particle_num ** 2)).mean()
    second_item = (second_item.sum(-1).sum(-1) / (particle_num ** 2)).mean()
    third_item = (third_item.sum(-1).sum(-1) / (particle_num ** 2)).mean()
    loss = first_item + second_item - 2 * third_item

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if count % update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())
    return loss.item()


if __name__ == '__main__':
    gamma = 0.99
    learning_rate = 5e-4
    batch_size = 64
    capacity = 100000
    exploration = 100
    epsilon_init = 0.9
    epsilon_min = 0.05
    decay = 0.992
    episode = 10000
    particle_num = 32
    render = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed = 2020
    set_seed(seed)
    env = gym.make('CartPole-v0')
    env.seed(seed)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    eval_net = mmd_ddqn(observation_dim, action_dim, particle_num).to(device)
    target_net = mmd_ddqn(observation_dim, action_dim, particle_num).to(device)
    target_net.load_state_dict(eval_net.state_dict())
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)
    buffer = replay_buffer(capacity)
    epsilon = epsilon_init
    update_freq = 200
    count = 0

    weight_reward = None
    for i in range(episode):
        obs = env.reset()
        if epsilon > epsilon_min:
            epsilon = epsilon * decay
        reward_total = 0
        if render:
            env.render()
        total_loss = 0
        while True:
            action = eval_net.act(torch.FloatTensor(np.expand_dims(obs, 0)).to(device), epsilon)
            count += 1
            next_obs, reward, done, info = env.step(action)
            buffer.store(obs, action, reward, next_obs, done)
            reward_total += reward
            obs = next_obs
            if render:
                env.render()
            if i > exploration:
                loss = train(buffer, eval_net, target_net, gamma, optimizer, batch_size, count, update_freq, device)
                total_loss += loss

            if done:
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                print('episode: {}\tepsilon: {:.2f}\treward: {}\tweight_reward: {:.3f}'.format(i+1, epsilon, reward_total, weight_reward))
                break
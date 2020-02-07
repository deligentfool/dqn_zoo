import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import gym
from collections import deque

class n_step_replay_buffer(object):
    def __init__(self, capacity, n_step, gamma):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.memory = deque(maxlen=self.capacity)
        self.n_step_buffer = deque(maxlen=self.n_step)

    def get_n_step_info(self):
        observation, action = self.n_step_buffer[0][: 2]
        reward, next_observation, done = self.n_step_buffer[-1][-3:]
        for _, _, rew, next_obs, do in reversed(list(self.n_step_buffer)[: -1]):
            reward = self.gamma * reward * (1 - do) + rew
            next_observation, done = (next_obs, do) if do else (next_observation, done)
        return observation, action, reward, next_observation, done

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)

        self.n_step_buffer.append([observation, action, reward, next_observation, done])
        if len(self.n_step_buffer) < self.n_step:
            return

        observation, action, reward, next_observation, done = self.get_n_step_info()
        self.memory.append([observation, action, reward, next_observation, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observation, action, reward, next_observation, done = zip(* batch)

        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done

    def __len__(self):
        return len(self.memory)


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


class qr_dqn(nn.Module):
    def __init__(self, observation_dim, action_dim, quant_num):
        super(qr_dqn, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.quant_num = quant_num

        self.fc1 = nn.Linear(self.observation_dim, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, self.action_dim * self.quant_num)

    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def act(self, observation, epsilon):
        if random.random() > epsilon:
            dist = self.forward(observation)
            action = dist.view(-1, self.action_dim, self.quant_num).mean(2).max(1)[1].detach().item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action


def get_target_distribution(target_model, next_observation, reward, done, gamma, action_dim, quant_num):
    batch_size = next_observation.size(0)

    next_dist = target_model.forward(next_observation).detach()
    next_dist = next_dist.view(-1, action_dim, quant_num)
    next_action = next_dist.mean(2).max(1)[1].detach()
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, quant_num)
    next_dist = next_dist.gather(1, next_action).unsqueeze(1)

    reward = reward.unsqueeze(1).expand(batch_size, quant_num)
    done = done.unsqueeze(1).expand(batch_size, quant_num)
    target_dist = reward + gamma * (1 - done) * next_dist
    target_dist.detach_()

    quant_idx = torch.sort(next_dist, 1, descending=False)[1]
    tau_hat = torch.linspace(0.0, 1.0 - 1. / quant_num, quant_num) + 0.5 / quant_num
    tau_hat = tau_hat.unsqueeze(0).expand(batch_size, quant_num)
    batch_idx = np.arange(batch_size)
    tau = tau_hat[:, quant_idx][batch_idx, batch_idx]
    return target_dist, tau


def train(eval_model, target_model, buffer, optimizer, gamma, action_dim, quant_num, batch_size, count, update_freq, k=1.):
    observation, action, reward, next_observation, done = buffer.sample(batch_size)

    observation = torch.FloatTensor(observation)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    next_observation = torch.FloatTensor(next_observation)
    done = torch.FloatTensor(done)

    dist = eval_model.forward(observation)
    dist = dist.view(-1, action_dim, quant_num)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, quant_num)
    dist = dist.gather(1, action).unsqueeze(1)
    target_dist, tau = get_target_distribution(target_model, next_observation, reward, done, gamma, action_dim, quant_num)

    u = target_dist - dist

    huber_loss = 0.5 * u.abs().clamp(min=0., max=k).pow(2)
    huber_loss = huber_loss + k * (u.abs() - u.abs().clamp(min=0., max=k) - 0.5 * k)
    quantile_loss = (tau - (u < 0).float()).abs() * huber_loss
    loss = quantile_loss.sum() / batch_size

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(eval_model.parameters(), 0.5)
    optimizer.step()

    if count % update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())


if __name__ == '__main__':
    epsilon_init = 0.95
    epsilon_decay = 0.995
    epsilon_min = 0.01
    gamma = 0.99
    learning_rate = 1e-3
    capacity = 100000
    exploration = 200
    episode = 1000000
    quant_num = 10
    update_freq = 200
    n_step = 1
    batch_size = 64
    k = 1.
    render = False

    env = gym.make('CartPole-v0')
    env = env.unwrapped
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    #buffer = n_step_replay_buffer(capacity, n_step, gamma)
    buffer = replay_buffer(capacity)
    eval_net = qr_dqn(observation_dim, action_dim, quant_num)
    target_net = qr_dqn(observation_dim, action_dim, quant_num)
    target_net.load_state_dict(eval_net.state_dict())
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)
    count = 0
    epsilon = epsilon_init
    weight_reward = None

    for i in range(episode):
        obs = env.reset()
        reward_total = 0
        if render:
            env.render()
        while True:
            action = eval_net.act(torch.FloatTensor(np.expand_dims(obs, 0)), epsilon)
            next_obs, reward, done, info = env.step(action)
            buffer.store(obs, action, reward, next_obs, done)
            if render:
                env.render()
            reward_total += reward
            count = count + 1
            obs = next_obs
            if i > exploration:
                train(eval_net, target_net, buffer, optimizer, gamma, action_dim, quant_num, batch_size, count, update_freq, k=1.)
            if done:
                if epsilon > epsilon_min:
                    epsilon = epsilon * epsilon_decay
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                print('episode: {}  reward: {}  weight_reward: {:.3f}  epsilon: {:.2f}'.format(i+1, reward_total, weight_reward, epsilon))
                break
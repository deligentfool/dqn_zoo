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


    h_list = list(np.linspace(1, 400, 20))
    #loss = loss_fn(q_value, expected_q_value.detach())
    first_item = 0
    first_kernel = -(q_particle_value.unsqueeze(-1) - q_particle_value.unsqueeze(-2)).pow(2)
    for h in h_list:
        first_item += (first_kernel / h).exp()
    first_item = (first_item.sum(-1).sum(-1) / (particle_num ** 2)).mean()

    second_item = 0
    second_kernel = -(expected_q_particle_value.unsqueeze(-1) - expected_q_particle_value.unsqueeze(-2)).pow(2)
    for h in h_list:
        second_item += (second_kernel / h).exp()
    second_item = (second_item.sum(-1).sum(-1) / (particle_num ** 2)).mean()

    third_item = 0
    third_kernel = -(q_particle_value.unsqueeze(-1) - expected_q_particle_value.unsqueeze(-2)).pow(2)
    for h in h_list:
        third_item += (third_kernel / h).exp()
    third_item = (third_item.sum(-1).sum(-1) / (particle_num ** 2)).mean()
    loss = first_item + second_item - 2 * third_item

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if count % update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())


if __name__ == '__main__':
    gamma = 0.99
    learning_rate = 5e-4
    batch_size = 64
    capacity = 100000
    exploration = 100
    epsilon_init = 0.9
    epsilon_min = 0.01
    decay = 0.998
    episode = 10000
    particle_num = 200
    render = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = gym.make('CartPole-v0')
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    eval_net = mmd_ddqn(observation_dim, action_dim, particle_num).to(device)
    target_net = mmd_ddqn(observation_dim, action_dim, particle_num).to(device)
    target_net.load_state_dict(eval_net.state_dict())
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate, eps=0.0003125)
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
                train(buffer, eval_net, target_net, gamma, optimizer, batch_size, count, update_freq, device)

            if done:
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                print('episode: {}  epsilon: {:.2f}  reward: {}  weight_reward: {:.3f}'.format(i+1, epsilon, reward_total, weight_reward))
                break


import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import gym
from collections import deque

class n_step_replay_buffer(object):
    def __init__(self, capacity, n_step, gamma):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.memory = deque(maxlen=self.capacity)
        self.n_step_buffer = deque(maxlen=self.n_step)

    def _get_n_step_info(self):
        reward, next_observation, done = self.n_step_buffer[-1][-3:]
        for _, _, rew, next_obs, do in reversed(list(self.n_step_buffer)[: -1]):
            reward = self.gamma * reward * (1 - do) + rew
            next_observation, done = (next_obs, do) if do else (next_observation, done)
        return reward, next_observation, done

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)

        self.n_step_buffer.append([observation, action, reward, next_observation, done])
        if len(self.n_step_buffer) < self.n_step:
            return
        reward, next_observation, done = self._get_n_step_info()
        observation, action = self.n_step_buffer[0][: 2]
        self.memory.append([observation, action, reward, next_observation, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observation, action, reward, next_observation, done = zip(* batch)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done

    def __len__(self):
        return len(self.memory)


class ddqn(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(ddqn, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.observation_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.action_dim)

    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
    expected_q_value = reward + gamma * (1 - done) * next_q_value

    #loss = loss_fn(q_value, expected_q_value.detach())
    loss = (expected_q_value.detach() - q_value).pow(2)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if count % soft_update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())


if __name__ == '__main__':
    gamma = 0.99
    learning_rate = 1e-3
    batch_size = 64
    soft_update_freq = 200
    capacity = 10000
    exploration = 100
    epsilon_init = 0.9
    epsilon_min = 0.01
    decay = 0.99
    episode = 1000000
    render = False

    env = gym.make('CartPole-v0')
    env = env.unwrapped
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    target_net = ddqn(observation_dim, action_dim)
    eval_net = ddqn(observation_dim, action_dim)
    eval_net.load_state_dict(target_net.state_dict())
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)
    buffer = n_step_replay_buffer(capacity, 4, gamma)
    loss_fn = nn.MSELoss()
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
                train(buffer, target_net, eval_net, gamma, optimizer, batch_size, loss_fn, count, soft_update_freq)

            if done:
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                print('episode: {}  epsilon: {:.2f}  reward: {}  weight_reward: {:.3f}'.format(i+1, epsilon, reward_total, weight_reward))
                break


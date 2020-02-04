import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import gym
import cv2


def img_to_gray(img, out_shape):
    img = cv2.resize(src=img, dsize=out_shape)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY)
    img = cv2.normalize(src=img, dst=img, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = np.expand_dims(img, 0)
    return img


def obs_stack(img_buffer):
    observation = np.concatenate([img_buffer[i] for i in range(4)], 0)
    return observation


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


class ddqn(nn.Module):
    def __init__(self, action_dim):
        super(ddqn, self).__init__()
        self.action_dim = action_dim
        self.conv1 = nn.Conv2d(4, 8, 4, 4)
        self.conv2 = nn.Conv2d(8, 8, 2, 2)
        self.conv3 = nn.Conv2d(8, 8, 2, 1)
        self.fc1 = nn.Linear(4 * 4 * 8, 64)
        self.fc2 = nn.Linear(64, self.action_dim)

    def forward(self, observation):
        x = self.conv1(observation)
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
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
    expected_q_value = reward + gamma * (1 - done) * next_q_value

    loss = loss_fn(q_value, expected_q_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if count % soft_update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())


if __name__ == '__main__':
    gamma = 0.99
    learning_rate = 1e-3
    batch_size = 64
    soft_update_freq = 100
    capacity = 10000
    exploration = 5
    epsilon_init = 0.9
    epsilon_min = 0.05
    decay = 0.995
    episode = 1000000
    render = True

    env = gym.make('Pong-v0')
    env = env.unwrapped
    action_dim = env.action_space.n
    target_net = ddqn(action_dim)
    eval_net = ddqn(action_dim)
    eval_net.load_state_dict(target_net.state_dict())
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)
    buffer = replay_buffer(capacity)
    loss_fn = nn.MSELoss()
    epsilon = epsilon_init
    count = 0

    weight_reward = None
    for i in range(episode):
        img_buffer = deque(maxlen=4)
        obs = env.reset()
        if epsilon > epsilon_min:
            epsilon = epsilon * decay
        reward_total = 0
        obs = img_to_gray(img=obs, out_shape=(84, 84))
        img_buffer.extend([obs] * 4)
        observation = obs_stack(img_buffer)
        if render:
            env.render()
        while True:
            action = eval_net.act(torch.FloatTensor(np.expand_dims(observation, 0)), epsilon)
            count += 1
            next_obs, reward, done, info = env.step(action)
            if render:
                env.render()
            next_obs = img_to_gray(img=next_obs, out_shape=(84, 84))
            img_buffer.append(next_obs)
            next_observation = obs_stack(img_buffer)
            buffer.store(observation, action, reward, next_observation, done)
            reward_total += reward
            observation = next_observation
            if i > exploration:
                train(buffer, target_net, eval_net, gamma, optimizer, batch_size, loss_fn, count, soft_update_freq)

            if done:
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                print('episode: {}  epsilon: {:.2f}  reward: {}  weight_reward: {:.3f}'.format(i+1, epsilon, reward_total, weight_reward))
                break


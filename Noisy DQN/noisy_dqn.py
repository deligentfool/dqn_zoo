import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gym

class prioritized_replay_buffer(object):
    def __init__(self, capacity, alpha, beta, beta_increment):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.pos = 0
        self.memory = []
        self.priorities = np.zeros([self.capacity], dtype=np.float32)

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)

        max_prior = np.max(self.priorities) if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append([observation, action, reward, next_observation, done])
        else:
            self.memory[self.pos] = [observation, action, reward, next_observation, done]
        self.priorities[self.pos] = max_prior
        self.pos += 1
        self.pos = self.pos % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < self.capacity:
            probs = self.priorities[: len(self.memory)]
        else:
            probs = self.priorities
        probs = probs ** self.alpha
        probs = probs / np.sum(probs)

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probs[indices]) ** (- self.beta)
        if self.beta < 1:
            self.beta += self.beta_increment
        weights = weights / np.max(weights)
        weights = np.array(weights, dtype=np.float32)

        observation, action, reward, next_observation, done = zip(* samples)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)


class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))
        self.register_buffer('weight_epsilon', torch.FloatTensor(self.output_dim, self.input_dim))

        self.bias_mu = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(self.output_dim))

        self.reset_parameter()
        self.reset_noise()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

    def reset_parameter(self):
        mu_range = 1 / np.sqrt(self.input_dim)

        self.weight_mu.detach().uniform_(-mu_range, mu_range)
        self.bias_mu.detach().uniform_(-mu_range, mu_range)

        self.weight_sigma.detach().fill_(self.std_init / np.sqrt(self.input_dim))
        self.bias_sigma.detach().fill_(self.std_init / np.sqrt(self.output_dim))

    def _scale_noise(self, size):
        noise = torch.randn(size)
        noise = noise.sign().mul(noise.abs().sqrt())
        return noise

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))


class ddqn(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(ddqn, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.fc = nn.Linear(self.observation_dim, 128)
        self.noisy1 = NoisyLinear(128, 128)
        self.noisy2 = NoisyLinear(128, self.action_dim)

    def forward(self, observation):
        x = self.fc(observation)
        x = F.relu(x)
        x = self.noisy1(x)
        x = F.relu(x)
        x = self.noisy2(x)
        return x

    def act(self, observation, epsilon):
        if random.random() > epsilon:
            q_value = self.forward(observation)
            action = q_value.max(1)[1].detach()[0].item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()



def train(buffer, eval_model, target_model, batch_size, count, update_freq, gamma, optimizer):
    observation, action, reward, next_observation, done, indices, weights = buffer.sample(batch_size)

    observation = torch.FloatTensor(observation)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    next_observation = torch.FloatTensor(next_observation)
    done = torch.FloatTensor(done)
    weights = torch.FloatTensor(weights)

    q_values = eval_model.forward(observation)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_values = target_model.forward(next_observation)
    next_q_value = next_q_values.max(1)[0].detach()
    expected_q_value = reward + gamma * (1 - done) * next_q_value

    loss = (expected_q_value - q_value).pow(2) * weights
    priorities = loss + 1e-5
    priorities = priorities.detach().numpy()
    buffer.update_priorities(indices, priorities)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    eval_model.reset_noise()
    target_model.reset_noise()

    if count % update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())


if __name__ == '__main__':
    epsilon_init = 0.9
    epsilon_min = 0.005
    epsilon_decay = 0.99
    capacity = 10000
    exploration = 2000
    update_freq = 1000
    batch_size = 64
    episode = 1000000
    render = False
    learning_rate = 1e-3
    gamma = 0.99
    alpha = 0.6
    beta = 0.4
    beta_increment_step = 1000000


    env = gym.make('CartPole-v0')
    env = env.unwrapped
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    beta_increment = (1 - beta) / beta_increment_step
    target_net = ddqn(observation_dim, action_dim)
    eval_net = ddqn(observation_dim, action_dim)
    eval_net.load_state_dict(target_net.state_dict())
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)
    buffer = prioritized_replay_buffer(capacity, alpha=alpha, beta=beta, beta_increment=beta_increment)
    epsilon = epsilon_init
    count = 0

    weight_reward = None
    for i in range(episode):
        obs = env.reset()
        if epsilon > epsilon_min:
            epsilon = epsilon * epsilon_decay
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
            if len(buffer) > exploration:
                train(buffer, eval_net, target_net, batch_size, count, update_freq, gamma, optimizer)
            if done:
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                print('episode: {}  epsilon: {:.2f}  reward: {}  weight_reward: {:.3f}'.format(i+1, epsilon, reward_total, weight_reward))
                break

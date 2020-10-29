import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gym
import math
from collections import deque


def set_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


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


class psi_net(nn.Module):
    def __init__(self, observation_dim, embedding_dim, hidden_dim):
        super(psi_net, self).__init__()
        self.observation_dim = observation_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.fc_layer = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embedding_dim)
        )

    def forward(self, observation):
        return self.fc_layer(observation)


class phi_net(nn.Module):
    def __init__(self, embedding_dim, quantile_num, cosine_num):
        super(phi_net, self).__init__()
        self.embedding_dim = embedding_dim
        self.quantile_num = quantile_num
        self.cosine_num = cosine_num

        self.cosine_layer = nn.Sequential(
            nn.Linear(self.cosine_num, self.embedding_dim),
            nn.ReLU()
        )

    def forward(self, taus):
        factors = torch.arange(0, self.cosine_num, 1.0).unsqueeze(0).unsqueeze(0)
        cos_trans = torch.cos(factors * taus.unsqueeze(-1).detach() * np.pi)
        rand_feat = self.cosine_layer(cos_trans)
        return rand_feat


class f_net(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, action_dim):
        super(f_net, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.fc_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

    def forward(self, embedding):
        return self.fc_layer(embedding)


class g_net(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, action_dim):
        super(g_net, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.fc_layer = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.ReLU()
        )

    def forward(self, prod, diff):
        inputs = torch.cat([prod, diff], dim=-1)
        return self.fc_layer(inputs)


class fraction_net(nn.Module):
    def __init__(self, quantile_num, state_embedding_dim):
        super(fraction_net, self).__init__()
        self.quantile_num = quantile_num
        self.state_embedding_dim = state_embedding_dim

        self.layer = nn.Sequential(
            nn.Linear(self.state_embedding_dim, self.quantile_num),
            nn.Softmax(dim=-1)
        )

    def forward(self, state_embedding):
        assert not state_embedding.requires_grad
        q = self.layer(state_embedding.detach())
        return q


class ndqfn_net(nn.Module):
    def __init__(self, observation_dim, action_dim, quant_num, cosine_num, p_num=32, epsilon=0.001, hidden_dim=128):
        super(ndqfn_net, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.quant_num = quant_num
        self.cosine_num = cosine_num
        self.p_num = p_num
        self.epsilon = epsilon
        self.hidden_dim = hidden_dim
        self.embedding_dim = hidden_dim
        self.p = torch.arange(0., 1., 1. / self.p_num)
        self.p = torch.cat([self.p, torch.ones([1])], dim=-1)
        self.p[0] += self.epsilon
        self.p[-1] -= self.epsilon

        self.psi_net = psi_net(self.observation_dim, self.embedding_dim, self.hidden_dim)
        self.phi_net = phi_net(self.embedding_dim, self.quant_num, self.cosine_num)
        self.f_net = f_net(self.embedding_dim, self.hidden_dim, self.action_dim)
        self.g_net = g_net(self.embedding_dim, self.hidden_dim, self.action_dim)
        self.fraction_net = fraction_net(self.quant_num, self.embedding_dim)

    def calc_state_embedding(self, observation):
        return self.psi_net(observation)

    def calc_quantile_fraction(self, state_embedding):
        assert not state_embedding.requires_grad
        q = self.fraction_net(state_embedding.detach())
        tau_0 = torch.zeros(q.size(0), 1)
        tau = torch.cat([tau_0, q], dim=-1)
        tau = torch.cumsum(tau, dim=-1)
        entropy = torch.distributions.Categorical(probs=q).entropy()
        tau_hat = ((tau[:, :-1] + tau[:, 1:]) / 2.).detach()
        return tau, tau_hat, entropy

    def calc_fix_quantile_value(self, state_embedding):
        rand_feat = self.phi_net(self.p)
        base = self.f_net(state_embedding).unsqueeze(-1)
        prod = state_embedding.unsqueeze(1) * rand_feat[:, 1:]
        diff = (rand_feat[:, 1:] - rand_feat[:, : -1]).repeat([prod.size(0), 1, 1])
        p_value = self.g_net(prod, diff).transpose(1, 2)
        p_value = torch.cat([base, p_value], dim=-1)
        #p_value = torch.cumsum(p_value, dim=-1)
        return p_value

    def calc_quantile_value(self, tau, state_embedding):
        assert not tau.requires_grad
        p_value = self.calc_fix_quantile_value(state_embedding)
        cum_sum_p_value = torch.cumsum(p_value[:, :, 1:], dim=-1) / self.p_num
        cum_sum_p_value = cum_sum_p_value + p_value[:, :, 0].unsqueeze(-1)
        cum_sum_p_value = torch.cat([p_value[:, :, 0].unsqueeze(-1), cum_sum_p_value], dim=-1)
        p_floor = (tau * self.p_num).floor().long()
        p_ceil = (tau * self.p_num).ceil().long()
        assert p_ceil.max() < p_value.size(-1)
        value_ceil = p_value.gather(2, p_ceil.unsqueeze(1).repeat([1, cum_sum_p_value.size(1), 1]))
        value = cum_sum_p_value.gather(2, p_floor.unsqueeze(1).repeat([1, cum_sum_p_value.size(1), 1]))
        #assert torch.min(self.p[p_ceil] - self.p[p_floor]) > 0
        value = value + ((tau - self.p[p_floor]) / torch.clamp_min(self.p[p_ceil] - self.p[p_floor], 1e-10) * ((self.p[p_ceil] - self.p[p_floor]) == 0)).unsqueeze(1).repeat([1, cum_sum_p_value.size(1), 1]) * value_ceil / self.p_num
        return value

    def act(self, observation, epsilon):
        if random.random() > epsilon:
            state_embedding = self.calc_state_embedding(observation)
            tau, tau_hat, _ = self.calc_quantile_fraction(state_embedding.detach())
            q_value = self.calc_q_value(state_embedding)
            action = q_value.max(1)[1].detach().item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action

    def calc_sa_quantile_value(self, state_embedding, action, tau):
        sa_quantile_value = self.calc_quantile_value(tau.detach(), state_embedding)
        sa_quantile_value = sa_quantile_value.gather(1, action.unsqueeze(-1).unsqueeze(-1).expand(sa_quantile_value.size(0), 1, sa_quantile_value.size(-1))).squeeze(1)
        return sa_quantile_value

    def calc_q_value(self, state_embedding):
        p_value = self.calc_fix_quantile_value(state_embedding)
        p_value = torch.cumsum(p_value, dim=-1)
        median_p = ((self.p[1:] - self.p[:-1]) / 2).unsqueeze(0).unsqueeze(0)
        q_value = ((p_value[:, :, 1:] + p_value[:, :, : -1]) * median_p).sum(-1)
        return q_value



class ndqfn(object):
    def __init__(self, env, capacity, episode, exploration, k, gamma, quant_num, cosine_num, batch_size, value_learning_rate, fraction_learning_rate, entropy_weight, epsilon_init, double_q, decay, epsilon_min, update_freq, render):
        self.env = env
        self.capacity = capacity
        self.episode = episode
        self.exploration = exploration
        self.k = k
        self.gamma = gamma
        self.batch_size = batch_size
        self.value_learning_rate = value_learning_rate
        self.fraction_learning_rate = fraction_learning_rate
        self.epsilon_init = epsilon_init
        self.decay = decay
        self.quant_num = quant_num
        self.epsilon_min = epsilon_min
        self.entropy_weight = entropy_weight
        self.update_freq = update_freq
        self.render = render
        self.cosine_num = cosine_num
        self.double_q = double_q

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.net = ndqfn_net(self.observation_dim, self.action_dim, self.quant_num, self.cosine_num)
        self.target_net = ndqfn_net(self.observation_dim, self.action_dim, self.quant_num, self.cosine_num)
        self.target_net.load_state_dict(self.net.state_dict())
        self.buffer = replay_buffer(self.capacity)
        self.quantile_value_param = list(self.net.psi_net.parameters()) + list(self.net.phi_net.parameters()) + list(self.net.f_net.parameters()) + list(self.net.g_net.parameters())
        self.quantile_fraction_param = list(self.net.fraction_net.parameters())
        self.quantile_value_optimizer = torch.optim.Adam(self.quantile_value_param, lr=self.value_learning_rate)
        self.quantile_fraction_optimizer = torch.optim.RMSprop(self.quantile_fraction_param, lr=self.fraction_learning_rate)
        self.epsilon = lambda x: self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(-1. * x / self.decay)
        self.count = 0
        self.weight_reward = None

    def calc_quantile_value_loss(self, tau, value, target_value):
        # * calculate quantile value loss
        # * get the quantile huber loss
        assert not tau.requires_grad
        u = target_value.unsqueeze(1) - value.unsqueeze(-1)
        huber_loss = 0.5 * u.abs().clamp(min=0., max=self.k).pow(2)
        huber_loss = huber_loss + self.k * (u.abs() - u.abs().clamp(min=0., max=self.k) - 0.5 * self.k)
        quantile_loss = (tau.unsqueeze(-1) - (u < 0).float()).abs() * huber_loss
        loss = quantile_loss.mean()
        return loss

    def calc_quantile_fraction_loss(self, observations, actions, tau, tau_hat):
        # * calculate quantile fraction loss
        assert not tau_hat.requires_grad
        sa_quantile_hat = self.net.calc_sa_quantile_value(observations, actions, tau_hat).detach()
        sa_quantile = self.net.calc_sa_quantile_value(observations, actions, tau[:, 1:-1]).detach()
        #gradient_tau = 2 * sa_quantile - sa_quantile_hat[:, :-1] - sa_quantile_hat[:, 1:]
        value_1 = sa_quantile - sa_quantile_hat[:, :-1]
        signs_1 = sa_quantile > torch.cat([sa_quantile_hat[:, :1], sa_quantile[:, :-1]], dim=-1)
        value_2 = sa_quantile - sa_quantile_hat[:, 1:]
        signs_2 = sa_quantile < torch.cat([sa_quantile[:, 1:], sa_quantile_hat[:, -1:]], dim=-1)
        gradient_tau = (torch.where(signs_1, value_1, -value_1) + torch.where(signs_2, value_2, -value_2)).view(*value_1.size())
        loss = (gradient_tau.detach() * tau[:, 1: -1]).sum(1).mean()
        return loss


    def train(self):
        observations, actions, rewards, next_observations, dones = self.buffer.sample(self.batch_size)

        observations = torch.FloatTensor(observations)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_observations = torch.FloatTensor(next_observations)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        state_embedding = self.net.calc_state_embedding(observations)
        tau, tau_hat, entropy = self.net.calc_quantile_fraction(state_embedding.detach())
        # * use tau_hat to calculate the quantile value
        dist = self.net.calc_quantile_value(tau_hat.detach(), state_embedding)
        value = dist.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(self.batch_size, 1, dist.size(2))).squeeze()

        if not self.double_q:
            next_state_embedding = self.target_net.calc_state_embedding(next_observations)
            # * have to use the eval_net's quantile fraction network
            next_tau, next_tau_hat, _ = self.net.calc_quantile_fraction(next_state_embedding.detach())
            target_actions = self.target_net.calc_q_value(next_state_embedding).max(1)[1].detach()
        else:
            next_state_embedding = self.net.calc_state_embedding(next_observations)
            next_tau, next_tau_hat, _ = self.net.calc_quantile_fraction(next_state_embedding.detach())
            target_actions = self.net.calc_q_value(next_state_embedding).max(1)[1].detach()
        next_state_embedding = self.target_net.calc_state_embedding(next_observations)
        # * also use tau_hat to calculate the quantile value
        target_dist = self.target_net.calc_quantile_value(tau_hat.detach(), next_state_embedding)
        target_value = target_dist.gather(1, target_actions.unsqueeze(1).unsqueeze(2).expand(self.batch_size, 1, target_dist.size(2))).squeeze()
        target_value = rewards + self.gamma * target_value * (1. - dones)
        target_value = target_value.detach()

        qauntile_value_loss = self.calc_quantile_value_loss(tau_hat.detach(), value, target_value)
        quantile_fraction_loss = self.calc_quantile_fraction_loss(state_embedding.detach(), actions, tau, tau_hat)
        entropy_loss = - (self.entropy_weight * entropy).mean()

        self.quantile_fraction_optimizer.zero_grad()
        quantile_fraction_loss.backward(retain_graph=True)
        #nn.utils.clip_grad_norm_(self.quantile_fraction_param, 10)
        self.quantile_fraction_optimizer.step()

        self.quantile_value_optimizer.zero_grad()
        qauntile_value_loss.backward()
        #nn.utils.clip_grad_norm_(self.quantile_value_param, 10)
        self.quantile_value_optimizer.step()

        if self.count % self.update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def run(self):
        for i in range(self.episode):
            obs = self.env.reset()
            if self.render:
                self.env.render()
            total_reward = 0
            while True:
                epsilon = self.epsilon(self.count)
                action = self.net.act(torch.FloatTensor(np.expand_dims(obs, 0)), epsilon)
                next_obs, reward, done, info = self.env.step(action)
                self.count += 1
                total_reward += reward
                if self.render:
                    self.env.render()
                self.buffer.store(obs, action, reward, next_obs, done)
                obs = next_obs

                if self.count > self.exploration:
                    self.train()

                if done:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward
                    print('episode: {}  reward: {}  weight_reward: {:.2f}  epsilon: {:.2f}'.format(i + 1, total_reward, self.weight_reward, epsilon))
                    break


if __name__ == '__main__':
    seed = 2020
    set_seed(seed)
    env = gym.make('CartPole-v0')
    #env = env.unwrapped
    env.seed(seed)
    test = ndqfn(
        env=env,
        capacity=10000,
        episode=100000,
        exploration=1000,
        k=1.,
        gamma=0.99,
        batch_size=32,
        quant_num=32,
        cosine_num=64,
        value_learning_rate=1e-3,
        fraction_learning_rate=1e-9,
        entropy_weight=0,
        double_q=True,
        epsilon_init=1,
        decay=5000,
        epsilon_min=0.01,
        update_freq=200,
        render=False
    )
    test.run()
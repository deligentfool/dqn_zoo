import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import gym


def set_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


#def calc_bandwidth(first_kernel, second_kernel, third_kernel, kernel_num=20, max_scale=200, min_scale=1):
#    # * kernel: [batch_size, particle_num, particle_num]
#    particle_num = first_kernel.size(-1)
#    bandwidth_list = list(np.linspace(min_scale, max_scale, kernel_num))
#    first_items = 0
#    third_items = 0
#    for h in bandwidth_list:
#        first_inner_distance = (-first_kernel / h).exp()
#        intra_distance = (-third_kernel / h).exp()
#
#        first_items += first_inner_distance
#        third_items += intra_distance
#
#    return first_items, third_items


def calc_bandwidth(first_kernel, third_kernel, kernel_num=20):
    # * kernel: [batch_size, particle_num, particle_num]
    particle_num = first_kernel.size(-1)
    bandwidth_list = [2 ** i for i in range(kernel_num)]
    first_items = 0
    third_items = 0
    for h in bandwidth_list:
        h = 2 * (h ** 2)
        first_inner_distance = (-first_kernel / h).exp()
        intra_distance = (-third_kernel / h).exp()

        first_items += first_inner_distance
        third_items += intra_distance

    return first_items, third_items


#def calc_bandwidth(first_kernel, second_kernel, third_kernel, kernel_num=20, max_scale=1.0, min_scale=0.01):
#    # * kernel: [batch_size, particle_num, particle_num]
#    kernel_mean = third_kernel.mean(-1).max(-1)[0]
#    particle_num = first_kernel.size(-1)
#    scale_list = list(np.linspace(min_scale, max_scale, num=kernel_num))
#    bandwidth_list = [(kernel_mean * scale).view(-1, 1, 1).detach() for scale in scale_list]
#    first_items = 0
#    third_items = 0
#    for h in bandwidth_list:
#        first_inner_distance = (-first_kernel / h).exp()
#        intra_distance = (-third_kernel / h).exp()
#
#        first_items += first_inner_distance
#        third_items += intra_distance
#
#    return first_items, third_items


#def calc_bandwidth(first_kernel, second_kernel, third_kernel, kernel_num=20, topk=10, max_scale=2.5, min_scale=1.5):
#    # * kernel: [batch_size, particle_num, particle_num]
#    kernel_median = third_kernel.median(-1)[0].median(-1)[0]
#    particle_num = first_kernel.size(-1)
#    scale_list = list(np.linspace(min_scale, max_scale, num=kernel_num))
#    bandwidth_list = [(kernel_median * scale).view(-1, 1, 1).detach() for scale in scale_list]
#    true_counts = []
#    first_items = []
#    third_items = []
#    for h in bandwidth_list:
#        first_inner_distance = (-first_kernel / h).exp()
#        second_inner_distance = (-second_kernel / h).exp()
#        intra_distance = (-third_kernel / h).exp()
#
#        first_true_count = (first_inner_distance.mean(-1) > intra_distance.mean(-1)).sum(-1, keepdim=True)
#        second_true_count = (second_inner_distance.mean(-1) > intra_distance.mean(-2)).sum(-1, keepdim=True)
#        true_count = first_true_count + second_true_count
#        true_counts.append(true_count)
#        first_items.append(first_inner_distance)
#        third_items.append(intra_distance)
#
#    _, idxs = torch.cat(true_counts, dim=-1).detach().topk(topk)
#    first_items = torch.stack(first_items, dim=-3)
#    third_items = torch.stack(third_items, dim=-3)
#    idxs = idxs.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, first_items.size(-2), first_items.size(-1)])
#    first_items = first_items.gather(-3, idxs).sum(-3)
#    third_items = third_items.gather(-3, idxs).sum(-3)
#    return first_items, third_items


#def calc_bandwidth(first_kernel, second_kernel, third_kernel, kernel_num=40, topk=20, max_scale=200, min_scale=1):
#    # * kernel: [batch_size, particle_num, particle_num]
#    particle_num = first_kernel.size(-1)
#    #bandwidth_list = list(min_scale + np.random.rand(kernel_num) * (max_scale - min_scale))
#    bandwidth_list = list(np.linspace(min_scale, max_scale, kernel_num))
#    true_counts = []
#    first_items = []
#    third_items = []
#    for h in bandwidth_list:
#        first_inner_distance = (-first_kernel / h).exp()
#        second_inner_distance = (-second_kernel / h).exp()
#        intra_distance = (-third_kernel / h).exp()
#
#        first_true_count = (first_inner_distance.mean(-1) > intra_distance.mean(-1)).sum(-1, keepdim=True)
#        second_true_count = (second_inner_distance.mean(-1) > intra_distance.mean(-2)).sum(-1, keepdim=True)
#        true_count = first_true_count + second_true_count
#        true_counts.append(true_count)
#        first_items.append(first_inner_distance)
#        third_items.append(intra_distance)
#
#    _, idxs = torch.cat(true_counts, dim=-1).detach().topk(topk)
#    first_items = torch.stack(first_items, dim=-3)
#    third_items = torch.stack(third_items, dim=-3)
#    idxs = idxs.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, first_items.size(-2), first_items.size(-1)])
#    first_items = first_items.gather(-3, idxs).sum(-3)
#    third_items = third_items.gather(-3, idxs).sum(-3)
#    return first_items, third_items

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


    first_kernel = (q_particle_value.unsqueeze(-1) - q_particle_value.unsqueeze(-2)).pow(2)
    #second_kernel = (expected_q_particle_value.unsqueeze(-1) - expected_q_particle_value.unsqueeze(-2)).pow(2)
    third_kernel = (q_particle_value.unsqueeze(-1) - expected_q_particle_value.unsqueeze(-2)).pow(2)
    first_item, third_item = calc_bandwidth(first_kernel, third_kernel)
    first_item = (first_item.sum(-1).sum(-1) / (particle_num ** 2)).mean()
    third_item = (third_item.sum(-1).sum(-1) / (particle_num ** 2)).mean()
    loss = first_item - 2 * third_item

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
    epsilon_min = 0.05
    decay = 0.992
    episode = 10000
    particle_num = 32
    render = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(2020)
    env = gym.make('CartPole-v0')
    env.seed(2020)
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


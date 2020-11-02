from utils import *
from torch.distributions import Normal
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from env import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PPOModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=64, hidden_dim2=32):
        super(PPOModel, self).__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, action_dim),
            nn.Tanh()
        )
        self.actor_var = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, action_dim),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x):
        raise NotImplementedError

    def act(self, x):
        return self.actor_mean(x), self.actor_var(x)

    def evaluate(self, x):
        return self.critic(x)


class PPOAgent(object):
    def __init__(self, state_dim, action_dim, memory, args):
        self.gamma = args.gamma
        self.eps_clip = args.eps_clip
        self.n_updates = args.n_updates
        self.policy = PPOModel(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.policy_old = PPOModel(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.criterion = nn.MSELoss()
        self.memory = memory

    def select_action(self, state):
        state_ = torch.from_numpy(state).to(device)
        # calculate the action based on the old policy wrt \theta_old
        action_mean, action_var = self.policy_old.act(state_)
        dist = Normal(action_mean, action_var)
        action = torch.clamp(dist.sample(), -1, 1 - state[0])
        action_logprob = dist.log_prob(action)
        # restore the state, actions, log_prob into the memory
        self.memory.states.append(state_)
        self.memory.actions.append(action)
        self.memory.logprobs.append(action_logprob)
        return action.detach().cpu().data.numpy().flatten()

    def evaluate_action(self, state, action):
        # calculate the action based on the policy wrt \theta
        action_mean, action_var = self.policy.act(state)
        dist = Normal(action_mean, action_var)
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.policy.evaluate(state)
        return action_logprob, torch.squeeze(state_value), dist_entropy

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.done)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = ((rewards - rewards.mean()) / (rewards.std() + 1e-5))

        old_states = torch.squeeze(torch.stack(self.memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(self.memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs).to(device), 1).detach()

        for i in range(self.n_updates):
            logprobs, state_values, dist_entropy = self.evaluate_action(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.criterion(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    parser = argparse.ArgumentParser(description='PyTorch on WP2.2')
    parser.add_argument('--max_episodes', default=20000, type=int)
    parser.add_argument('--update_per_episodes', default=20, type=int)
    parser.add_argument('--log_per_episodes', default=20, type=int)
    parser.add_argument('--save_per_episodes', default=500, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--n_updates', default=5, type=int)

    parser.add_argument('--eps_clip', default=0.2, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    args = parser.parse_args()

    env = make_env()
    agent = PPOAgent(env.observation_space, env.action_space, memory=Memory(), args=args)
    train(agent, env, args)
    test(agent, env)


if __name__ == '__main__':
    main()


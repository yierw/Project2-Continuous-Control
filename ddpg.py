import torch
import random
import numpy as np

from model import Critic, Actor
from OUNoise import OUNoise

def soft_update(target, source, tau):
    """ Perform soft update"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Agent:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.SmoothL1Loss()
    iter = 0
    def __init__(self, o_dim, a_dim, lr_actor = 1e-3, lr_critic = 1e-3,
                 batch_size = 16, gamma = 0.99, tau = 0.001, buffer_size = int(1e5),
                 seed = 1234):

        self.o_dim = o_dim
        self.a_dim = a_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.seed = seed

        # Replay memory
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.seed)

        # Actor Network (w/ Target Network)
        self.actor = Actor(o_dim, a_dim).to(self.device)
        self.target_actor = Actor(o_dim, a_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = lr_actor)

        # Critic Network (w/ Target Network)
        self.critic = Critic(o_dim, a_dim).to(self.device)
        self.target_critic = Critic(o_dim, a_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = lr_critic, weight_decay = 0.0)

        # Noise process
        self.noise = OUNoise(a_dim)


    def get_action(self, state, eps = 0.):
        """
        action value ranges from -1 to 1
        --
        eps = 0. no exploration
            > 0. add exploration
        """
        state_tensor = torch.FloatTensor(state).view(1, self.o_dim).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor)[0].detach().cpu().numpy()
        self.actor.train()
        action += self.noise.sample() * eps
        return np.clip(action, -1, 1)

    def update(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).view(-1, 1).to(self.device)
        dones = torch.FloatTensor(dones).view(-1, 1).to(self.device)
        # ---------------------------- update critic ---------------------------- #
        next_actions = self.target_actor(next_states)
        Q_next = self.target_critic(next_states, next_actions)
        Q_targets = rewards + self.gamma * Q_next * (1. -dones)
        Q_expected = self.critic(states, actions)
        critic_loss = self.loss_fn(Q_expected, Q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_targets(self):
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

    def step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        if (len(self.buffer) > self.batch_size):
            experiences = self.buffer.sample()
            self.update(experiences)
            self.update_targets()
            self.iter += 1

    def reset(self):
        self.noise.reset()

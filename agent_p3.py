import numpy as np
import random
from collections import namedtuple, deque

from actor_critic_p3 import Actor, Critic
from utils_p3 import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPGAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size=24, action_size=2, num_agents=2, seed=0, gamma=0.99, 
                 tau=1e-3, lr_actor=1e-4, lr_critic=1e-3, weight_decay_actor=0, weight_decay_critic=0, 
                 epsilon=1.0, epsilon_decay=1e-6, clip_grad=1.0):
        super(MADDPGAgent, self).__init__()
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.clip_grad = clip_grad

        # Actor Network (w/ Target Network)
        self.actor = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        # self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=weight_decay_actor)

        self.hard_update(self.actor, self.actor_target)

        # Critic Network (w/ Target Network)
        self.critic = Critic(state_size, action_size, seed, num_agents).to(device)
        self.critic_target = Critic(state_size, action_size, seed, num_agents).to(device)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay_critic)

        self.hard_update(self.critic, self.critic_target)

        # Noise process
        self.noise = OUNoise(action_size, seed)

    def hard_update(self, local_model, target_model):
        """Copy weights from local to target network (one-time initialization)."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)        

    def act(self, state, noise=True):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            noise(bool): add Ornstein-Uhlenbeck noise
        """
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float()
        state = state.to(device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()

        # Add noise for exploration
        if noise:
            action += self.epsilon * self.noise.sample()
        
        return np.clip(action, -1, 1)

    def learn(self, samples, agent_id, all_agents):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Unpack data from replay buffer and convert to tensors
        states = torch.tensor([exp[0] for exp in samples], dtype=torch.float, device=device)
        actions = torch.tensor([exp[1] for exp in samples], dtype=torch.float, device=device)
        reward = torch.tensor([exp[2] for exp in samples], dtype=torch.float, device=device)
        next_states = torch.tensor([exp[3] for exp in samples], dtype=torch.float, device=device)
        done = torch.tensor([exp[4] for exp in samples], dtype=torch.float, device=device)
        all_states = torch.tensor([exp[5] for exp in samples], dtype=torch.float, device=device)
        all_next_states = torch.tensor([exp[6] for exp in samples], dtype=torch.float, device=device)
        all_actions = torch.tensor([exp[7] for exp in samples], dtype=torch.float, device=device)

        # Extract this agent's rewards and dones
        agent_reward = reward.view(-1, 1)  
        agent_done = done.view(-1, 1)      

        # ============================================================================
        # Update critic
        # ============================================================================    
        self.critic_optimizer.zero_grad()

        # Get next actions from all agents' TARGET actors
        with torch.no_grad():
            next_actions = []
            for i in range(self.num_agents):
                next_state_i = all_next_states[:, i, :]
                next_action_i = self.actor_target(next_state_i)
                next_actions.append(next_action_i)
            next_actions = torch.cat(next_actions, dim=1)  # [batch_size, num_agents * action_size]

        # Prepare concatenated states for critic
        critic_target_states = all_next_states.view(all_next_states.shape[0], -1)  # [batch_size, num_agents * state_size]
        
        # Compute target Q-values
        with torch.no_grad():
            Q_target_next = self.critic_target(critic_target_states, next_actions)
        Q_target = agent_reward + self.gamma * Q_target_next * (1 - agent_done)
        
        # Prepare current states and actions for critic
        critic_states = all_states.view(all_states.shape[0], -1)  # [batch_size, num_agents * state_size]
        critic_actions = all_actions.view(all_actions.shape[0], -1)  # [batch_size, num_agents * action_size]
        
        # Compute current Q-values
        Q_expected = self.critic(critic_states, critic_actions)

        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_target.detach())
        
        # Backpropagate
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad)
        self.critic_optimizer.step()

        # ============================================================================
        # Update actor
        # ============================================================================
        self.actor_optimizer.zero_grad()

        # Get actions from all agents' LOCAL actors (WITHOUT NOISE!)
        # FIX: Use self.actor, NOT self.act (which adds noise and clamps)
        actions_pred = []
        for i in range(self.num_agents):
            state_i = all_states[:, i, :]
            if i == agent_id:
                # This agent: gradients enabled (for this agent's actor update)
                action_i = self.actor(state_i)
            else:
                # Other agents: detach gradients (we're only updating THIS agent's actor)
                action_i = all_agents[i].actor(state_i).detach()
            actions_pred.append(action_i)
        
        actions_pred = torch.cat(actions_pred, dim=1)  # [batch_size, num_agents * action_size]

        # Compute actor loss (negative because we want to maximize Q)
        actor_loss = -self.critic(critic_states, actions_pred).mean()

        # Backpropagate
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimizer.step()

        # ============================================================================
        # Update target networks (soft update)
        # ============================================================================
        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)

        # ============================================================================
        # Update noise
        # ============================================================================
        self.epsilon = max(0.01, self.epsilon - self.epsilon_decay)  
        self.noise.reset()  

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def reset(self):
        self.noise.reset()
    
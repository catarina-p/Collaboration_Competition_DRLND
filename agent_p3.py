import numpy as np
import random
from collections import namedtuple, deque

from actor_critic_p2 import Actor, Critic
from noise_p3 import OrnsteinUhlenbeckProcess
from per_p3 import PrioritizedReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 0.5e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 20       # learning timestep interval
NUM_PASSES = 10         # number of learning passes
EPSILON = 1.0           # explore->exploit noise process added to act step
EPSILON_DECAY = 1e-6    # decay rate for noise process

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPGagent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.epsilon = EPSILON

        # Actor Network (w/ Target Network)
        self.actor = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic = Critic(action_size * num_agents, action_size * num_agents, seed).to(device)
        self.critic_target = Critic(state_size * num_agents, action_size * num_agents, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OrnsteinUhlenbeckProcess(action_size, seed)

        # Replay memory
        self.replay = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.counter = 0
    
    def step(self, state, action, reward, next_state, done, step_num):
        """Save experience in replay memory, and learn."""

        # Save experience in replay memory
        self.replay.add(state, action, reward, next_state, done)
        self.counter += 1
        
        # Learn every UPDATE_EVERY time steps.
        if self.counter >= BATCH_SIZE and step_num % UPDATE_EVERY == 0:
            for _ in range(NUM_PASSES):
                self.learn(GAMMA)  
            # self.counter = 0 # TO RECHECK ALGORITHM LATER
        
        # Update beta for PER at the end of the episode
        if done:
            self.replay.update_beta()

    def act(self, state, noise=True):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            noise(bool): add Ornstein-Uhlenbeck noise
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action_values = self.actor(state[0])
            # action_values = self.actor(state).squeeze(0) #Claude
        self.actor.train()

        # Add noise
        if noise:
            action_values += self.epsilon * torch.from_numpy(self.noise.sample()).to(action_values.device).type_as(action_values)
        return np.clip(action_values, -1, 1)

    def learn(self, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        #Claude
        # Get random sample and compute priority weights
        sample_result = self.replay.sample()
        
        # Check if we got valid samples
        if sample_result[0] is None:
            return
        states, actions, rewards, next_states, dones, priorities, indices = sample_result

        # # Get random sample and compute priority weights
        # states, actions, rewards, next_states, dones, priorities, indices = self.replay.sample()

        # Update critic ---------------------------------------------------------------#
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)
        ## Compute current Q targets
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        ## Get expected Q values from local critic
        Q_expected = self.critic(states, actions)
        ## Update replay buffer priorities
        self.replay.update_priorities(Q_expected, Q_targets, indices)
        ## Compute loss
        priority_weights = self.replay.compute_weights(np.array(priorities))
        critic_loss = self.replay.PER_loss(Q_expected, Q_targets, priority_weights)
        ## Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()  

        # Update actor ----------------------------------------------------------------#
        ## Predicted actions from actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        ## Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target ---------------------------------------------------------------#
        self.soft_update(self.critic, self.critic_target, TAU)
        self.soft_update(self.actor, self.actor_target, TAU)

        # Update noise ----------------------------------------------------------------#
        self.epsilon = max(0, self.epsilon - EPSILON_DECAY)
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
    

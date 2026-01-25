import numpy as np
from collections import namedtuple, deque
import random

from sumTree_p3 import SumTree

import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prioritized Replay parameters
# In the orignal paper, alpha~0.7 and beta_i~0.5
SMALL = 0.0001  #P(i)~(|TD_error|+SMALL)^\alpha
alpha = 0.7 #0.8     #P(i)~(|TD_error|+SMALL)^\alpha
beta_i = 0.5 #0.7    #w_i =(1/(N*P(i)))^\beta
beta_f = 1.
beta_update_steps = 1000

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.tree_memory = SumTree(buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
       
        self.alpha = alpha
        self.beta = beta_i
        self.step_beta = 1       

        # clipping [-1,1] is used in a 'custom loss function'
        self.p_max = 1.+SMALL #initial priority with max value in [-1,1]
        self.priorities = deque(maxlen = buffer_size) #Importance sampling weights

        # Track number of experiences added (Claude)
        self.n_entries = 0
        self.buffer_size = buffer_size
    
    def add(self, states, actions, rewards, next_states, dones):
        """
        states:      (num_agents, state_size)
        actions:     (num_agents, action_size)
        rewards:     (num_agents,)
        next_states: (num_agents, state_size)
        dones:       (num_agents,)
        """
        e = (states, actions, rewards, next_states, dones)
        self.tree_memory.add(self.p_max, e)
        self.n_entries = min(self.n_entries + 1, self.buffer_size)

    
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = []
        priorities = []
        indices = []
        segment = self.tree_memory.total()/self.batch_size #total error

        for i in range(self.batch_size):
            a = segment*i
            b = segment*(i+1)
            rd = np.round(np.random.uniform(a,b),6) #should fix some problems with idx
            idx, priority, data = self.tree_memory.get(rd)
            # experiences.append( data + (priority,) + (idx,) )

            # Skip if data is not valid (happens when buffer not full) (Claude)
            if data is None or not isinstance(data, tuple):
                continue
            
            experiences.append( data + (priority,) + (idx,) )
            # experiences.append(data)
            # priorities.append(priority)
            # indices.append(idx)

        # # Check if we have enough valid experiences
        # if len(experiences) == 0:
        #     return None, None, None, None, None, None, None

        states = torch.from_numpy(np.stack([e[0] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.stack([e[1] for e in experiences])).float().to(device)
        rewards = torch.from_numpy(np.stack([e[2] for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.stack([e[3] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.stack([e[4] for e in experiences]).astype(np.uint8)).float().to(device)
        
        priorities = np.array(priorities)
        indices = np.array(indices)

        return states, actions, rewards, next_states, dones, priorities, indices

    # def PER_loss(self, input, target, weights):
    #     #Custom loss: prioritized replay introduces a bias,
    #     #             corrected with the importance-sampling weights.
    #     #input: input -- Q
    #     #       target -- r + gamma*Qhat(s', argmax_a' Q(s',a'))
    #     #       weights -- importance sampling weights
    #     #output:loss -- unbiased loss

    #     with torch.no_grad():
    #         tw = torch.tensor(weights).detach().float().to(device)

    #     loss = torch.clamp((input-target),-1,1)
    #     loss = loss**2
    #     loss = torch.sum(tw*loss)
    #     return loss
    
    # Claude
    def PER_loss(self, Q_expected, Q_target, weights):
        """Calculate PER loss with importance sampling weights.
        
        Args:
            Q_expected: Current Q-values from critic
            Q_targets: Target Q-values 
            weights: Importance sampling weights
            
        Returns:
            Weighted loss for backpropagation
        """
        # Convert weights to tensor if needed
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32, device=device)
        
        # Ensure weights have the right shape [batch_size, 1]
        if weights.dim() == 1:
            weights = weights.unsqueeze(1)
        
        # Calculate element-wise TD errors
        # td_errors = Q_target - Q_expected
        
        # Use Huber loss (smooth L1) which is more stable than MSE
        # This is equivalent to: 
        # - squared loss for small errors (|error| < 1)
        # - linear loss for large errors (|error| >= 1)
        element_wise_loss = F.smooth_l1_loss(Q_expected, Q_target.detach(), reduction='none')
        
        # Apply importance sampling weights
        weighted_loss = weights * element_wise_loss
        
        # Return mean weighted loss
        return weighted_loss.mean()

    
    def update_beta(self):
        # linearly increasing from beta_i~0.5 to beta_f = 1
        # self.beta = (beta_f-beta_i)*(self.step_beta-1)/(beta_update_steps-1) + beta_i
        #Claude
        self.step_beta += 1
        self.beta = min(beta_f, (beta_f-beta_i)*(self.step_beta-1)/(beta_update_steps-1) + beta_i)

    # def compute_weights(self, priorities):
    #     #compute importance sampling weight, before the update
    #     self.priorities.append(priorities)
    #     self.p_max = np.max(self.priorities)
    #     weights = (np.sum(self.priorities)/(len(self.priorities)*priorities)) #.reshape(-1,1)**self.beta
    #     weights /= self.p_max
    #     return weights

    def compute_weights(self, priorities):
        """Compute importance sampling weights for PER.

        Args:
            priorities: List or array of priorities for sampled experiences
            
        Returns:
            Normalized importance sampling weights
        """
        # Convert to numpy array if needed
        priorities = np.array(priorities)

        # Avoid division by zero
        priorities = np.maximum(priorities, 1e-8)

        # Compute sampling probabilities
        total_priority = self.tree_memory.total()
        if total_priority == 0:
            # If no priorities yet, return uniform weights
            return np.ones(len(priorities))

        probs = priorities / total_priority

        # Use actual number of stored experiences
        N = self.n_entries

        # Importance-sampling weights: w_i = (N * P(i))^(-beta)
        weights = np.power(N * probs, -self.beta)

        # Normalize weights by dividing by max weight for stability
        weights = weights / weights.max()

        return weights


    def update_priorities(self, Qexpected, Qtarget, indices):
        # with torch.no_grad():
        #         p = torch.abs(Qtarget - Qexpected)
        #         p = (p.cpu().numpy()+ SMALL)**alpha
        #         for j, idx in enumerate(indices):
        #             self.tree_memory.update(idx, p[j][0])
        #Claude
        """Update priorities in the SumTree based on TD errors.
        
        Args:
            Qexpected: Current Q-values from critic
            Qtarget: Target Q-values
            indices: Indices of experiences in the SumTree
        """
        with torch.no_grad():
            # Calculate TD errors
            td_errors = torch.abs(Qtarget - Qexpected)
            
            # Convert to numpy and flatten if needed
            td_errors = td_errors.cpu().numpy()
            if td_errors.ndim > 1:
                td_errors = td_errors.flatten()
            
            # Calculate new priorities: P(i) = (|TD_error| + ε)^α
            new_priorities = np.power(td_errors + SMALL, alpha)
            
            # Update each priority in the tree
            for idx, priority in zip(indices, new_priorities):
                self.tree_memory.update(idx, priority)
    
    # def __len__(self):
    #     """Return the current size of internal memory."""
    #     return len(self.tree_memory)

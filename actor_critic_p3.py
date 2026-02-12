import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layer(layer):
    in_feat = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(in_feat)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, states_size, act_size, seed, hidden_fcl = [128, 64, 32]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_fcl[0] (int): Number of nodes in first hidden layer
            hidden_fcl[1] (int): Number of nodes in second hidden layer
            hidden_fcl[2] (int): Number of nodes in third hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(states_size, hidden_fcl[0])
        # self.bn1 = nn.BatchNorm1d(hidden_fcl[0])
        self.fc2 = nn.Linear(hidden_fcl[0], hidden_fcl[1])
        self.fc3 = nn.Linear(hidden_fcl[1], hidden_fcl[2])
        self.fc4 = nn.Linear(hidden_fcl[2], act_size)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*init_layer(self.fc1))
        self.fc2.weight.data.uniform_(*init_layer(self.fc2))
        self.fc3.weight.data.uniform_(*init_layer(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))
    
class Critic(nn.Module):
    """Critic (Policy) Model."""

    def __init__(self, states_size, act_size, seed, num_agents=2, hidden_fcl=[128, 64, 32]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_fcl[0] (int): Number of nodes in first hidden layer
            hidden_fcl[1] (int): Number of nodes in second hidden layer
            hidden_fcl[2] (int): Number of nodes in third hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(states_size*num_agents, hidden_fcl[0])
        # self.bn1 = nn.BatchNorm1d(hidden_fcl[0])
        self.fc2 = nn.Linear(hidden_fcl[0], hidden_fcl[1])
        self.fc3 = nn.Linear(hidden_fcl[1] + act_size*num_agents, hidden_fcl[2])
        self.fc4 = nn.Linear(hidden_fcl[2], 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*init_layer(self.fc1))
        self.fc2.weight.data.uniform_(*init_layer(self.fc2))
        self.fc3.weight.data.uniform_(*init_layer(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, all_states, all_act):
        x = F.relu(self.fc1(all_states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(torch.cat([x, all_act], dim=1)))
        return self.fc4(x)


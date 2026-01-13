"""
A2C Neural Network Architectures

Actor-Critic Architecture:
- Actor: Outputs action probabilities (policy network)
- Critic: Estimates state values (value network)

"""

import torch
import torch.nn as nn
import numpy as np


def orthogonal_init(layer, gain=1.0):
    # Apply orthogonal initialization to a linear layer.

    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class ActorNet(nn.Module):

    # Actor Network (Policy Network)


    def __init__(self, state_dim, action_dim):
        
        super().__init__()

        # Build network layers
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

        # Apply orthogonal initialization
        orthogonal_init(self.fc1, gain=np.sqrt(2))
        orthogonal_init(self.fc2, gain=np.sqrt(2))
        # Use smaller gain for output layer to start with smaller action preferences
        orthogonal_init(self.fc3, gain=0.01)

    def forward(self, x):
    
        # Forward pass through the network.
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class CriticNet(nn.Module):

    # Critic Network (Value Network)

    def __init__(self, state_dim):

        super().__init__()

        # Build network layers
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        # Apply orthogonal initialization
        orthogonal_init(self.fc1, gain=np.sqrt(2))
        orthogonal_init(self.fc2, gain=np.sqrt(2))
        # Use gain=1.0 for value output
        orthogonal_init(self.fc3, gain=1.0)

    def forward(self, x):
        
        # Forward pass through the network.
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

"""
A2C Neural Network Architectures

This module defines the Actor and Critic networks used by the A2C agent.

Actor-Critic Architecture:
- Actor: Outputs action probabilities (policy network)
- Critic: Estimates state values (value network)

Both networks share a similar structure but serve different purposes:
- Actor learns WHAT to do (action selection)
- Critic learns HOW GOOD a state is (value estimation)
"""

import torch.nn as nn


class ActorNet(nn.Module):
    """
    Actor Network (Policy Network)

    Maps states to action logits (unnormalized probabilities).
    The output is passed through softmax to get actual probabilities.

    Architecture: state_dim -> 128 -> 128 -> action_dim
    """

    def __init__(self, state_dim, action_dim):
        """
        Initialize the Actor network.

        Args:
            state_dim: Number of input features (observation size)
            action_dim: Number of possible actions
        """
        super().__init__()  # Required for PyTorch modules

        # Simple feedforward network with ReLU activations
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),   # Input layer: state -> 128 neurons
            nn.ReLU(),                    # Non-linear activation
            nn.Linear(128, 128),          # Hidden layer: 128 -> 128 neurons
            nn.ReLU(),                    # Non-linear activation
            nn.Linear(128, action_dim)    # Output layer: 128 -> action logits
            # Note: No softmax here - it's applied in the agent's act() method
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Action logits of shape (batch_size, action_dim) or (action_dim,)
        """
        return self.net(x)


class CriticNet(nn.Module):
    """
    Critic Network (Value Network)

    Maps states to a single value estimate V(s).
    This value represents the expected cumulative reward from this state.

    Architecture: state_dim -> 128 -> 128 -> 1
    """

    def __init__(self, state_dim):
        """
        Initialize the Critic network.

        Args:
            state_dim: Number of input features (observation size)
        """
        super().__init__()  # Required for PyTorch modules

        # Simple feedforward network with ReLU activations
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),   # Input layer: state -> 128 neurons
            nn.ReLU(),                    # Non-linear activation
            nn.Linear(128, 128),          # Hidden layer: 128 -> 128 neurons
            nn.ReLU(),                    # Non-linear activation
            nn.Linear(128, 1)             # Output layer: single value V(s)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Value estimate of shape (batch_size, 1) or (1,)
        """
        return self.net(x)

import torch
import torch.nn as nn

class ActorNet(nn.Module):
    """
    Actor Network - larger architecture for better learning

    Architecture: 13 -> 256 -> 256 -> 5
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # Initialize weights with orthogonal initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


class CriticNet(nn.Module):
    """
    Critic Network - estimates state value V(s)

    Architecture: 13 -> 256 -> 256 -> 1
    """
    def __init__(self, state_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)

import torch.nn as nn                 

# Define a neural network class for Q-function (predicts Q-values for each action)
class QNetwork(nn.Module):
    # Constructor: state_dim = number of state features, action_dim = number of possible actions
    def __init__(self, state_dim=8, action_dim=5):
        super().__init__()
        # Simple MLP: state_dim -> 128 -> 128 -> action_dim
        # Output is Q-values for each action (NOT probabilities)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    # Forward function: how the input passes through the network
    def forward(self, x):
        # x: tensor of shape [state_dim] or [batch_size, state_dim]
        # returns Q-values: shape [action_dim] or [batch_size, action_dim]
        return self.net(x)

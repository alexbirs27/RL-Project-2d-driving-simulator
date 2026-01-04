import torch.nn as nn                   # components for defining networks (Linear, ReLU, etc.)

# Define a neural network class for the policy (actor)
class PolicyNet(nn.Module):     
    # Constructor: state_dim = number of state features, action_dim = number of possible actions
    def __init__(self, state_dim=0, action_dim=0):   
        # Simple MLP: state_dim -> 256 -> action_dim
        # Output is logits (NOT softmax). Softmax will be applied later in agent.act()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )


     # Forward function: how the input passes through the network
    def forward(self, x):              
        # x: tensor of shape [state_dim] or [batch_size, state_dim]
        # returns logits: shape [action_dim] or [batch_size, action_dim]
        return self.policy(x)



# Define a neural network for the value function (critic)
class ValueNet(nn.Module):    
    # Constructor: receives only state_dim (input = state)
    def __init__(self, state_dim=0):   
        super().__init__()

        # Simple MLP: state_dim -> 256 -> 1
        self.value = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    # Forward pass for the critic
    def forward(self, x):               
        # returns V(s): shape [1] or [batch_size, 1]
        return self.value(x)

import torch.nn as nn                   # components for defining networks (Linear, ReLU, etc.)

# Define a neural network class for the policy (actor)
class PolicyNet(nn.Module):     
    # Constructor: state_dim = number of state features, action_dim = number of possible actions
    def __init__(self, state_dim=0, action_dim=0):   
        # TODO: Define network layers (e.g., Linear, ReLU)
        pass

     # Forward function: how the input passes through the network
    def forward(self, x):              
        # TODO: Implement forward pass
        pass



# Define a neural network for the value function (critic)
class ValueNet(nn.Module):    
    # Constructor: receives only state_dim (input = state)
    def __init__(self, state_dim=0):   
        # TODO: Define network layers
        pass

    # Forward pass for the critic
    def forward(self, x):               
        # TODO: Implement forward pass
        pass

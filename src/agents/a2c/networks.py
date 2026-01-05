import torch.nn as nn

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__() # Obligatoriu!
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class CriticNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__() # Obligatoriu!
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Output: o singură valoare (Value)
        )

    def forward(self, x):
        return self.net(x)
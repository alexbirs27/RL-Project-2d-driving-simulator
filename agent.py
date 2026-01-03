import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple


class ActorCritic(nn.Module):
    """
    Neural network with shared layers and separate heads for actor and critic.
    
    Actor: outputs action probabilities
    Critic: outputs state value estimate
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # Shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Returns:
            action_probs: probability distribution over actions
            value: estimated state value
        """
        features = self.shared(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Returns:
            action: selected action
            log_prob: log probability of the action
            value: estimated state value
        """
        action_probs, value = self.forward(x)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        
        log_prob = torch.log(action_probs.squeeze(0)[action] + 1e-8)
        
        return action.item(), log_prob, value


class A2CAgent:
    """
    Advantage Actor-Critic (A2C) agent.
    
    Uses the advantage function to reduce variance in policy gradient estimation.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize network
        self.network = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Training statistics
        self.train_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """Select an action given a state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor, deterministic)
        
        return action, log_prob.item(), value.item()
    
    def compute_returns(self, rewards: list, values: list, dones: list) -> list:
        """
        Compute discounted returns using the value function for bootstrapping.
        """
        returns = []
        R = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                R = 0
            R = rewards[i] + self.gamma * R
            returns.insert(0, R)
        
        return returns
    
    def train_step(
        self,
        states: list,
        actions: list,
        rewards: list,
        values: list,
        log_probs: list,
        dones: list
    ) -> dict:
        """
        Perform one training step using collected trajectories.
        
        Returns:
            Dictionary with loss statistics
        """
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        
        # Compute returns
        returns = self.compute_returns(rewards, values, dones)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Forward pass
        action_probs, state_values = self.network(states)
        state_values = state_values.squeeze()
        
        # Calculate advantages
        advantages = returns - state_values.detach()
        
        # Policy loss (actor)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        policy_loss = -(new_log_probs * advantages).mean()
        
        # Value loss (critic)
        value_loss = nn.MSELoss()(state_values, returns)
        
        # Entropy bonus (encourage exploration)
        entropy = dist.entropy().mean()
        
        # Total loss
        total_loss = (
            policy_loss +
            self.value_coef * value_loss -
            self.entropy_coef * entropy
        )
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Record statistics
        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
        
        for key, value in stats.items():
            self.train_stats[key].append(value)
        
        return stats
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_stats': self.train_stats
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_stats = checkpoint['train_stats']
        print(f"Model loaded from {path}")
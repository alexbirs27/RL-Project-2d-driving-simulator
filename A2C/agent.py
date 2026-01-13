"""
A2C (Advantage Actor-Critic) Agent

A policy gradient reinforcement learning algorithm that uses:
1. Actor: Neural network that learns the policy (what action to take)
2. Critic: Neural network that estimates state values (how good a state is)

Key concepts:
- Advantage = Actual Return - Estimated Value
  If positive: action was better than expected -> increase probability
  If negative: action was worse than expected -> decrease probability

- The advantage reduces variance in policy gradient updates compared
  to using raw returns, leading to more stable training.

Reference: Mnih et al. (2016) - "Asynchronous Methods for Deep Reinforcement Learning"
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .networks import ActorNet, CriticNet


class A2CAgent:
    """
    Advantage Actor-Critic agent with separate actor and critic networks.

    The actor outputs action probabilities, the critic outputs state values.
    Both are trained together to maximize expected cumulative reward.
    """

    def __init__(self, state_dim=8, action_dim=5, lr=1e-3, gamma=0.99, entropy_coef=0.01):
        """
        Initialize the A2C agent.

        Args:
            state_dim: Number of features in the observation (e.g., 10)
            action_dim: Number of possible actions (e.g., 9)
            lr: Learning rate for both actor and critic optimizers
            gamma: Discount factor for future rewards (0.99 = long-term focus)
            entropy_coef: Weight for entropy bonus (encourages exploration)
        """
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # GPU support - automatically use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"A2C using device: {self.device}")

        # Initialize neural networks and move to device
        self.actor = ActorNet(state_dim, action_dim).to(self.device)
        self.critic = CriticNet(state_dim).to(self.device)

        # Separate optimizers for actor and critic
        # This allows different learning dynamics if needed
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)

    def act(self, state):
        """
        Select an action during training (with exploration).

        Uses the actor network to get action probabilities, then samples
        from that distribution. Returns both the action and its log probability
        (needed for the policy gradient update).

        Args:
            state: Current observation array of shape (state_dim,)

        Returns:
            action: Integer index of the selected action
            log_prob: Log probability of the selected action (for training)
        """
        # Convert numpy array to PyTorch tensor
        state_t = torch.tensor(state, dtype=torch.float32).to(self.device)

        # Get action logits from actor network
        logits = self.actor(state_t)

        # Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=-1)

        # Create a categorical distribution and sample from it
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def select_action(self, state):
        """
        Select an action during evaluation (deterministic, no exploration).

        Uses the actor network to get action probabilities, then picks
        the action with the highest probability (greedy selection).

        Args:
            state: Current observation array of shape (state_dim,)

        Returns:
            action: Integer index of the best action
        """
        state_t = torch.tensor(state, dtype=torch.float32).to(self.device)

        # No gradient computation needed during evaluation
        with torch.no_grad():
            logits = self.actor(state_t)
            probs = F.softmax(logits, dim=-1)
            action = torch.argmax(probs).item()

        return action

    def update(self, rewards, log_probs, states, dones, next_state):
        """
        Update both actor and critic networks using collected experience.

        This is where the actual learning happens (backpropagation).

        Args:
            rewards: List of rewards received at each step
            log_probs: List of log probabilities of actions taken
            states: List of states visited
            dones: List of done flags (True if episode ended)
            next_state: The final state reached

        Returns:
            total_loss: Combined loss value for logging
        """
        # Convert lists to tensors
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        log_probs_t = torch.stack(log_probs).to(self.device)
        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)

        # === Step 1: Compute Discounted Returns (R_t) ===
        # Returns are computed backwards from the last state
        # R_t = r_t + gamma * R_{t+1}
        returns = []

        # Bootstrap from the value of the last state (unless episode ended)
        R = self.critic(torch.tensor(next_state, dtype=torch.float32).to(self.device)).item()

        # Work backwards through the episode
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0  # Reset return at episode boundaries
            R = r + self.gamma * R
            returns.insert(0, R)

        returns_t = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize returns for training stability
        # This keeps the gradient magnitudes consistent across episodes
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # === Step 2: Compute Value Estimates V(s) ===
        # The critic predicts how good each state is
        values = self.critic(states_t).squeeze()

        # === Step 3: Compute Advantage ===
        # Advantage = Actual Return - Predicted Value
        # Tells us if the action was better or worse than expected
        advantage = returns_t - values

        # === Step 4: Compute Losses ===

        # Critic Loss: Mean Squared Error between predicted and actual returns
        # We want V(s) to accurately predict the returns
        critic_loss = F.mse_loss(values, returns_t)

        # Actor Loss: Policy gradient with advantage
        # Negative because we want to MAXIMIZE reward (gradient ascent)
        # advantage.detach() prevents gradients from flowing through the critic
        actor_loss = -(log_probs_t * advantage.detach()).mean()

        # Entropy Loss: Encourages exploration by penalizing certainty
        # Higher entropy = more uniform action distribution = more exploration
        logits = self.actor(states_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        entropy_loss = -dist.entropy().mean()  # Negative because we want to maximize entropy

        # Combine all losses
        # 0.5 weight on critic loss is a common choice
        total_loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss

        # === Step 5: Backpropagation ===

        # Zero out old gradients
        self.opt_actor.zero_grad()
        self.opt_critic.zero_grad()

        # Compute gradients
        total_loss.backward()

        # Gradient clipping for stability
        # Prevents exploding gradients that can destabilize training
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

        # Update network weights
        self.opt_actor.step()
        self.opt_critic.step()

        return total_loss.item()

    def save(self, path):
        """
        Save both actor and critic networks to a file.

        Args:
            path: File path to save the model checkpoint
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path):
        """
        Load actor and critic networks from a file.

        Args:
            path: File path to load the model checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

        # Set to evaluation mode (disables dropout, batch norm updates, etc.)
        self.actor.eval()
        self.critic.eval()

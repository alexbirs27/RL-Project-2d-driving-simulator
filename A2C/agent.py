import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .networks import ActorNet, CriticNet


class A2CAgent:

    def __init__(self, state_dim=8, action_dim=5, lr=1e-3, gamma=0.99, entropy_coef=0.01,
                 gae_lambda=0.95, total_episodes=2000):
    
        # Initialize the agent

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.total_episodes = total_episodes

        # Initial hyperparameters
        self.initial_lr = lr
        self.initial_entropy_coef = entropy_coef
        self.current_lr = lr
        self.current_entropy_coef = entropy_coef

        # Track current episode for decay scheduling
        self.current_episode = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"A2C using device: {self.device}")

        # Initialize neural networks and move to device
        self.actor = ActorNet(state_dim, action_dim).to(self.device)
        self.critic = CriticNet(state_dim).to(self.device)

        # Separate optimizers for actor and critic
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)

    def _update_learning_rate(self):
        
        # Update learning rate with linear decay.

        progress = min(self.current_episode / self.total_episodes, 1.0)
        # Linear decay to 10% of initial value
        self.current_lr = self.initial_lr * (1.0 - 0.9 * progress)

        # Update optimizer learning rates
        for param_group in self.opt_actor.param_groups:
            param_group['lr'] = self.current_lr
        for param_group in self.opt_critic.param_groups:
            param_group['lr'] = self.current_lr

    def _update_entropy_coef(self):
        
        # Update entropy coefficient with linear decay.
        progress = min(self.current_episode / self.total_episodes, 1.0)
        self.current_entropy_coef = self.initial_entropy_coef * (1.0 - 0.9 * progress)

    def act(self, state):
        
        # Select an action during training

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
        # Select an action during evaluation (no exploration)

        state_t = torch.tensor(state, dtype=torch.float32).to(self.device)

        # No gradient computation needed during evaluation
        with torch.no_grad():
            logits = self.actor(state_t)
            probs = F.softmax(logits, dim=-1)
            action = torch.argmax(probs).item()

        return action

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE formula: A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        """
        advantages = []
        gae = 0

        # Convert values to list for easier indexing
        values_list = values.detach().cpu().numpy().flatten().tolist()
        values_list.append(next_value)

        # Compute GAE backwards through the episode
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values_list[t]
                gae = delta  # Reset GAE at episode boundaries
            else:
                delta = rewards[t] + self.gamma * values_list[t + 1] - values_list[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + values.squeeze()

        return advantages, returns

    def update(self, rewards, log_probs, states, dones, next_state):
        # Update both actor and critic networks using collected experience.
        
        # Update decay schedules
        self._update_learning_rate()
        self._update_entropy_coef()
        self.current_episode += 1

        # Convert lists to tensors
        log_probs_t = torch.stack(log_probs).to(self.device)
        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)

        # Compute Value Estimates V(s)
        values = self.critic(states_t).squeeze()

        # Get value of the next state for bootstrapping
        with torch.no_grad():
            next_value = self.critic(
                torch.tensor(next_state, dtype=torch.float32).to(self.device)
            ).item()

        # Compute GAE Advantages and Returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # Normalize advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute Losses

        # Critic Loss
        critic_loss = F.mse_loss(values, returns.detach())

        # Actor Loss
        # Negative because we want to MAXIMIZE reward (gradient ascent)
        actor_loss = -(log_probs_t * advantages.detach()).mean()

        # Entropy Loss
        logits = self.actor(states_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        entropy_loss = -dist.entropy().mean()
        total_loss = actor_loss + 0.5 * critic_loss + self.current_entropy_coef * entropy_loss

        # Backpropagation

        # Zero out old gradients
        self.opt_actor.zero_grad()
        self.opt_critic.zero_grad()

        # Compute gradients
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

        # Update network weights
        self.opt_actor.step()
        self.opt_critic.step()

        return total_loss.item()

    def save(self, path):
        
        # Save both actor and critic networks

        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'current_episode': self.current_episode,
            'current_lr': self.current_lr,
            'current_entropy_coef': self.current_entropy_coef
        }, path)

    def load(self, path):
        # Load actor and critic networks
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

        # Restore training state if available
        if 'current_episode' in checkpoint:
            self.current_episode = checkpoint['current_episode']
        if 'current_lr' in checkpoint:
            self.current_lr = checkpoint['current_lr']
        if 'current_entropy_coef' in checkpoint:
            self.current_entropy_coef = checkpoint['current_entropy_coef']

        # Set to evaluation mode (disables dropout, batch norm updates, etc.)
        self.actor.eval()
        self.critic.eval()

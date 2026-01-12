import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .networks import ActorNet, CriticNet

class A2CAgent:

    def __init__(self, state_dim=13, action_dim=5, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, entropy_coef=0.01, max_grad_norm=0.5):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.initial_entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.initial_lr = lr

        # Initializare retele
        self.actor = ActorNet(state_dim, action_dim)
        self.critic = CriticNet(state_dim)

        # Optimizatoare
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)

    def update_learning_rate(self, current_episode, total_episodes):
        """Linear decay: lr goes from initial_lr to 10% of initial_lr"""
        progress = current_episode / total_episodes
        new_lr = self.initial_lr * (0.1 + 0.9 * (1.0 - progress))

        for param_group in self.opt_actor.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.opt_critic.param_groups:
            param_group['lr'] = new_lr

    def update_entropy_coef(self, current_episode, total_episodes):
        """Decay entropy coefficient from initial value to 10%"""
        progress = current_episode / total_episodes
        self.entropy_coef = self.initial_entropy_coef * (0.1 + 0.9 * (1.0 - progress))

    def act(self, state):
        """
        Primeste starea si returneaza actiunea + log_probabilitatea ei.
        Folosit in timpul antrenamentului.
        """
        state_t = torch.tensor(state, dtype=torch.float32)

        # Actor
        logits = self.actor(state_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def select_action(self, state):
        """
        Versiune simplificata doar pentru evaluare (fara gradient).
        """
        state_t = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            logits = self.actor(state_t)
            probs = F.softmax(logits, dim=-1)
            action = torch.argmax(probs).item()
        return action

    def compute_gae(self, rewards, values, next_value, dones):
        """
        Generalized Advantage Estimation (GAE) - same as PPO
        This gives much better gradient estimates than simple returns.
        """
        values = values + [next_value]
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            done = 1.0 if dones[t] else 0.0

            # TD error: delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - done) - values[t]

            # GAE recursion: A_t = delta_t + gamma * lambda * (1-done) * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1.0 - done) * gae
            advantages.insert(0, gae)

        # Returns target for critic: R_t = A_t + V(s_t)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        return (
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
        )

    def update(self, rewards, log_probs, states, dones, next_state):
        """
        Improved update with GAE instead of simple discounted returns.
        """
        # Convert to tensors
        log_probs_t = torch.stack(log_probs)
        states_t = torch.tensor(np.array(states), dtype=torch.float32)

        # Get values for all states
        with torch.no_grad():
            values = self.critic(states_t).squeeze().tolist()
            if not isinstance(values, list):
                values = [values]
            next_value = self.critic(torch.tensor(next_state, dtype=torch.float32)).item()

        # Compute GAE advantages and returns
        advantages, returns_t = self.compute_gae(rewards, values, next_value, dones)

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Recompute values (with gradient this time)
        values_t = self.critic(states_t).squeeze()

        # Critic Loss: MSE between predicted values and returns
        critic_loss = F.mse_loss(values_t, returns_t)

        # Actor Loss: -log_prob * advantage
        actor_loss = -(log_probs_t * advantages.detach()).mean()

        # Entropy Loss: Bonus for exploration
        logits = self.actor(states_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        entropy_loss = -dist.entropy().mean()

        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss

        # Backpropagation
        self.opt_actor.zero_grad()
        self.opt_critic.zero_grad()
        total_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.opt_actor.step()
        self.opt_critic.step()

        return total_loss.item()

    def save(self, path):
        """Save model with optimizer states"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'opt_actor': self.opt_actor.state_dict(),
            'opt_critic': self.opt_critic.state_dict()
        }, path)

    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        if 'opt_actor' in checkpoint:
            self.opt_actor.load_state_dict(checkpoint['opt_actor'])
            self.opt_critic.load_state_dict(checkpoint['opt_critic'])
        self.actor.eval()
        self.critic.eval()

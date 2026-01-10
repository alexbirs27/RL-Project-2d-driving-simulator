import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .networks import ActorNet, CriticNet

class A2CAgent:
    def __init__(self, state_dim=8, action_dim=5, lr=1e-3, gamma=0.99, entropy_coef=0.01):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        # Inițializare rețele
        self.actor = ActorNet(state_dim, action_dim)
        self.critic = CriticNet(state_dim)
        
        # Optimizatoare
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)

    def act(self, state):
        """
        Primește starea și returnează acțiunea + log_probabilitatea ei.
        Folosit în timpul antrenamentului.
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
        Versiune simplificată doar pentru evaluare (fără gradient).
        """
        state_t = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            logits = self.actor(state_t)
            probs = F.softmax(logits, dim=-1)
            action = torch.argmax(probs).item()
        return action

    def update(self, rewards, log_probs, states, dones, next_state):
        """
        Aici se întâmplă învățarea (Backpropagation).
        """
        # Convertim totul la tensori
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        log_probs_t = torch.stack(log_probs)
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        
        # 1. Calculăm Discounted Returns (R_t)
        returns = []
        R = self.critic(torch.tensor(next_state, dtype=torch.float32)).item()
        
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns_t = torch.tensor(returns, dtype=torch.float32)
        
        # Normalizăm returns pentru stabilitate (opțional, dar recomandat)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # 2. Calculăm Values (V(s)) prezise de Critic
        values = self.critic(states_t).squeeze()
        
        # 3. Calculăm Advantage: A = Returns - Values
        # Advantage ne spune cât de bună a fost acțiunea față de medie
        advantage = returns_t - values
        
        # 4. Calculăm Loss-urile
        
        # Critic Loss: Vrem ca V(s) să fie cât mai aproape de Returns (MSE)
        critic_loss = F.mse_loss(values, returns_t)
        
        # Actor Loss: - (log_prob * advantage)
        # Dacă advantage e pozitiv, creștem probabilitatea. Dacă e negativ, o scădem.
        actor_loss = -(log_probs_t * advantage.detach()).mean()
        
        # Entropy Loss: Bonus pentru explorare (distribuție uniformă)
        # Calculăm entropia distribuției curente
        logits = self.actor(states_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        entropy_loss = -dist.entropy().mean()

        # Loss total
        total_loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss
        
        # 5. Backpropagation
        self.opt_actor.zero_grad()
        self.opt_critic.zero_grad()
        total_loss.backward()
        
        # Opțional: Gradient Clipping pentru stabilitate
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        
        self.opt_actor.step()
        self.opt_critic.step()
        
        return total_loss.item()

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor.eval()
        self.critic.eval()
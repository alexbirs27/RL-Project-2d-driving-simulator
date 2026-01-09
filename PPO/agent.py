import torch                            # framework for neural networks and tensor operations
import torch.optim as optim             # Import optimizers (Adam, SGD, etc.)
import torch.nn.functional as F
import numpy as np

from .networks import PolicyNet, ValueNet


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, gamma, lam, clip_eps, lr,
                 steps_per_epoch, train_iters, minibatch_size, entropy_coef, max_grad_norm):
        # All hyperparameters come from config.py

        #hyperparametrii
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.steps_per_epoch = steps_per_epoch
        self.train_iters = train_iters
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.initial_entropy_coef = entropy_coef  # Store initial value for decay
        self.max_grad_norm = max_grad_norm
        self.initial_lr = lr  

        # actor+critic
        self.policy = PolicyNet(state_dim, action_dim)  # Create actor network: receives state -> produces action logits
        self.value_fn = ValueNet(state_dim)             # Create critic network: receives state -> produces V(s)

        #optimizers
        self.opt_policy = optim.Adam(self.policy.parameters(), lr=lr) # Optimizer for policy (actor)
        self.opt_value = optim.Adam(self.value_fn.parameters(), lr=lr) # Optimizer for value function (critic)


    def update_learning_rate(self, current_epoch, total_epochs):
        """Linear decay: lr goes from initial_lr to 10% of initial_lr over training"""
        progress = current_epoch / total_epochs
        new_lr = self.initial_lr * (0.1 + 0.9 * (1.0 - progress))  # Floor at 10% of initial LR

        for param_group in self.opt_policy.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.opt_value.param_groups:
            param_group['lr'] = new_lr

    def update_entropy_coef(self, current_epoch, total_epochs):
        """Decay entropy coefficient from initial value to 10% over training (less exploration over time)"""
        progress = current_epoch / total_epochs
        self.entropy_coef = self.initial_entropy_coef * (0.1 + 0.9 * (1.0 - progress))

    def save_checkpoint(self, filepath, epoch, reward):
        """Save model checkpoint with policy, value network, and training info"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_fn.state_dict(),
            'policy_optimizer': self.opt_policy.state_dict(),
            'value_optimizer': self.opt_value.state_dict(),
            'epoch': epoch,
            'reward': reward
        }, filepath)

    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_fn.load_state_dict(checkpoint['value_state_dict'])
        self.opt_policy.load_state_dict(checkpoint['policy_optimizer'])
        self.opt_value.load_state_dict(checkpoint['value_optimizer'])
        return checkpoint['epoch'], checkpoint['reward']

    #action selection= Function that receives a state and returns an action + logp + value (without gradient)
    def act(self, state):  
        # input
        # state = numpy array 

        # output: 
        # action = int 
        # log probability of the action under current policy = tensor, detached
        # V(s) =tensor, detached 
        #                            
        #converting state => torch tensor
        s = torch.tensor(state, dtype=torch.float32)

        #actor forward: logits => probabilities
        logits = self.policy(s)                    # shape: [action_dim]
        probs = F.softmax(logits, dim=-1)          # shape: [action_dim]

        #discrete distribution over actions
        dist = torch.distributions.Categorical(probs)

        #sample action
        action = dist.sample()                     #tensor scalar
        logp = dist.log_prob(action)               #log π(a|s)
        value = self.value_fn(s)                   #V(s), shape: [1]

        #detach because rollout data is "old policy data"
        return action.item(), logp.detach(), value.detach()


    # Generalized Advantage Estimation (GAE) - Calculates advantage and returns for the entire rollout
    def compute_gae(self, rewards, values, next_value, dones): 
        # input:
        # rewards = list[float] length T 
        # values = list[float] length T (V(s_t) for each step)
        # next_value = float (V(s_{T}) for bootstrap) 
        # dones = list[bool] length T

        #output:
        # advantages = tensor shape [T] 
        # returns = tensor shape [T]

        #append bootstrap value so values[t+1] is valid at last step
        values = values + [next_value]

        advantages = []
        returns = []
        gae = 0.0
        #iterate backwards through time
        for t in reversed(range(len(rewards))):
            done = 1.0 if dones[t] else 0.0

            #TD error: δ_t = r_t + γ V(s_{t+1}) (1-done) - V(s_t)
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - done) - values[t]

            #GAE recursion: A_t = δ_t + γλ(1-done)A_{t+1}
            gae = delta + self.gamma * self.lam * (1.0 - done) * gae
            advantages.insert(0, gae)

        #Returns target for critic: R_t = A_t + V(s_t)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        return (
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
        ) #Convert to tensors


    # PPO update (the actual learning) - Receives the rollout (collected data) and performs update
    def train(self, obs, actions, logp_old, advantages, returns):  
        #input:
        # obs = tensor shape [N, state_dim]
        # actions = tensor shape [N] (inits)
        # logp_old = tensor shape [N] ((log π_old(a|s)))
        #badvantages=  tensor shape [N]
        
        #output:
        # tensor shape [N]
        dataset_size = len(obs)

        #normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.train_iters):
            idx = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, self.minibatch_size):
                batch_idx = idx[start : start + self.minibatch_size]

                b_obs = obs[batch_idx]                 # [B, state_dim]
                b_actions = actions[batch_idx]         # [B]
                b_logp_old = logp_old[batch_idx]       # [B]
                b_adv = advantages[batch_idx]          # [B]
                b_returns = returns[batch_idx]         # [B]

                #Recompute log probs under current policy
                logits = self.policy(b_obs)            # [B, action_dim]
                probs = F.softmax(logits, dim=-1)      # [B, action_dim]
                dist = torch.distributions.Categorical(probs)

                new_logp = dist.log_prob(b_actions)    # [B]

                # Ratio: π_new(a|s) / π_old(a|s)
                ratio = torch.exp(new_logp - b_logp_old)

                # Clipped ratio
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

                #entropy bonus
                entropy = dist.entropy().mean()

                #Policy loss (PPO clipped objective)
                policy_loss = -torch.min(ratio * b_adv, clipped * b_adv).mean() - self.entropy_coef * entropy

                #Value loss
                value_pred = self.value_fn(b_obs).squeeze(-1)  # [B]
                value_loss = (b_returns - value_pred).pow(2).mean()

                #Optimize actor
                self.opt_policy.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt_policy.step()

                #Optimize critic
                self.opt_value.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_fn.parameters(), self.max_grad_norm)
                self.opt_value.step()

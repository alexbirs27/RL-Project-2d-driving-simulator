import torch                            # framework for neural networks and tensor operations
import torch.optim as optim             # Import optimizers (Adam, SGD, etc.)

from .networks import PolicyNet, ValueNet


# PPO Agent
class PPOAgent:                         
    def __init__(self,                  # Constructor with PPO hyperparameters
                 state_dim=0,           # State dimension (how many numbers are in the observation)
                 action_dim=0,          # Number of possible discrete actions
                 gamma=0.99,            # Discount factor: how much the future matters (0.99 = future matters a lot)
                 lam=0.95,              # Lambda for GAE: control between bias/variance
                 clip_eps=0.2,          # Epsilon for PPO clipping (how much the policy is allowed to change per update)
                 lr=3e-4,               # Learning rate: how large the learning steps are
                 steps_per_epoch=4096,  # How many steps to collect in the environment before an update
                 train_iters=10,        # How many times to pass through the data during update (internal epochs on the collected batch)
                 minibatch_size=64):    # Minibatch size during training

        self.gamma = gamma             
        self.lam = lam                 
        self.clip_eps = clip_eps       
        self.steps_per_epoch = steps_per_epoch
        self.train_iters = train_iters        
        self.minibatch_size = minibatch_size  

        self.policy = PolicyNet(state_dim, action_dim)  # Create actor network: receives state -> produces action logits
        self.value_fn = ValueNet(state_dim)             # Create critic network: receives state -> produces V(s)

        self.opt_policy = optim.Adam(self.policy.parameters(), lr=lr) # Optimizer for policy (actor)
        self.opt_value = optim.Adam(self.value_fn.parameters(), lr=lr) # Optimizer for value function (critic)




    # Action selection - Function that receives a state and returns an action + logp + value (without gradient)
    def act(self, state):                                
        # TODO: Implement action selection logic
        return action, logp, value  


    # Generalized Advantage Estimation (GAE) - Calculates advantage and returns for the entire rollout
    def compute_gae(self, rewards, values, next_value, dones): 
        # TODO: Implement GAE calculation
        advantages = []
        returns = []
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)  # Convert to tensors


    # PPO update (the actual learning) - Receives the rollout (collected data) and performs update
    def train(self, obs, actions, logp_old, advantages, returns):  
        # TODO: Implement PPO clipped objective and value function update
        pass

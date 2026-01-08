import torch                            # framework for neural networks and tensor operations
import torch.optim as optim             # Import optimizers (Adam, SGD, etc.)
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

from .networks import QNetwork


# Replay Buffer - stores past experiences for training
class ReplayBuffer:
    def __init__(self, capacity=10000):
        # deque: efficient FIFO queue with maxlen (removes oldest when full)
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store a single transition (s, a, r, s', done)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample a batch of transitions
        # Returns: lists of states, actions, rewards, next_states, dones
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# DQN Agent
class DQNAgent:
    def __init__(self,
                 state_dim=8,           # State dimension: 5 rays + centerline + velocity
                 action_dim=9,          # Number of possible discrete actions
                 gamma=0.99,            # Discount factor: how much the future matters
                 lr=1e-3,               # Learning rate
                 buffer_size=10000,     # Replay buffer capacity
                 batch_size=64,         # Minibatch size for training
                 epsilon_start=1.0,     # Initial exploration rate
                 epsilon_end=0.01,      # Final exploration rate
                 epsilon_decay=1000,    # Epsilon decay steps (exponential)
                 tau=0.005):            # Soft update rate for target network

        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.action_dim = action_dim
        self.steps_done = 0
        self.initial_lr = lr  # Store initial learning rate for decay

        # Q-networks: online (learning) and target (stable targets)
        self.q_network = QNetwork(state_dim, action_dim)      # Online network - updated every step
        self.target_network = QNetwork(state_dim, action_dim) # Target network - updated with soft updates
        self.target_network.load_state_dict(self.q_network.state_dict()) # Initialize target = online

        # Optimizer for Q-network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)


    # Action selection using epsilon-greedy strategy with exponential decay
    def act(self, state, training=True):
        # input: state = numpy array
        # output: action = int

        import math

        # Calculate epsilon threshold using exponential decay
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        # Exploration: random action with probability epsilon
        if training and random.random() < eps_threshold:
            return random.randint(0, self.action_dim - 1)

        # Exploitation: choose action with highest Q-value
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            q_values = self.q_network(state_tensor)  # Forward pass
            action = torch.argmax(q_values).item()   # Pick best action

        return action


    # Store transition in replay buffer
    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)


    # Training step - sample from buffer and update Q-network
    def train(self):
        # Only train if we have enough samples in buffer
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample a random minibatch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)           # [batch_size, state_dim]
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)        # [batch_size, 1]
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)      # [batch_size, 1]
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32) # [batch_size, state_dim]
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)          # [batch_size, 1]

        # Compute current Q-values: Q(s, a)
        current_q_values = self.q_network(states).gather(1, actions)  # [batch_size, 1]

        # Compute target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]  # [batch_size, 1]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)      # [batch_size, 1]

        # Compute loss (Huber/SmoothL1 loss - more robust than MSE)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()

        # Soft update of target network (polyak averaging)
        self.soft_update_target_network()

        return loss.item()


    # Soft update of target network using polyak averaging
    # θ_target = τ * θ_online + (1 - τ) * θ_target
    def soft_update_target_network(self):
        target_state_dict = self.target_network.state_dict()
        policy_state_dict = self.q_network.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key] * self.tau + target_state_dict[key] * (1 - self.tau)
        self.target_network.load_state_dict(target_state_dict)

    # Hard update target network (copy weights from online network) - for backwards compatibility
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Get current epsilon value for logging
    def get_epsilon(self):
        import math
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)

    # Deprecated - epsilon now decays automatically in act()
    def decay_epsilon(self):
        pass

    def update_learning_rate(self, current_episode, total_episodes):
        """
        Decay learning rate linearly to prevent catastrophic forgetting.

        As training progresses, smaller learning rates help stabilize the policy
        and prevent large gradient updates from destroying learned behavior.

        Args:
            current_episode: Current training episode
            total_episodes: Total number of training episodes
        """
        progress = current_episode / total_episodes
        # Decay from initial_lr to 1% of initial_lr (100x reduction)
        new_lr = self.initial_lr * (0.01 + 0.99 * (1.0 - progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        return new_lr

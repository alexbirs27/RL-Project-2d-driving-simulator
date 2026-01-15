import numpy as np

class ARSAgent:

    def __init__(self, state_dim, action_dim, learning_rate=0.03, noise=0.025,
                 num_deltas=32, num_best_deltas=16, normalize_states=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.noise = noise
        self.initial_noise = noise
        self.num_deltas = num_deltas
        self.num_best_deltas = num_best_deltas  # Top-k deltas to use
        self.normalize_states = normalize_states

        # Linear policy: Weight matrix
        self.weights = np.zeros((action_dim, state_dim))

        # Running statistics for state normalization
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        self.state_n = 0  # Number of states seen

    def update_state_stats(self, states):
        states = np.array(states)
        batch_mean = np.mean(states, axis=0)
        batch_std = np.std(states, axis=0)
        batch_n = len(states)

        # Welford's online algorithm for running mean/variance
        new_n = self.state_n + batch_n
        delta = batch_mean - self.state_mean

        self.state_mean = self.state_mean + delta * batch_n / new_n

        # Update variance (simplified)
        if self.state_n > 0:
            self.state_std = (self.state_std * self.state_n + batch_std * batch_n) / new_n
        else:
            self.state_std = batch_std

        # Prevent division by zero
        self.state_std = np.maximum(self.state_std, 1e-6)
        self.state_n = new_n

    def normalize_state(self, state):
        if self.normalize_states and self.state_n > 0:
            return (state - self.state_mean) / self.state_std
        return state

    def select_action(self, state, delta=None, direction=None):

        # Normalize state
        norm_state = self.normalize_state(state)

        weights = self.weights.copy()
        if delta is not None:
            if direction == "plus":
                weights += delta * self.noise
            else:
                weights -= delta * self.noise

        # Inmultire matriceala: greutati * observatii
        logits = weights.dot(norm_state)
        # Alegem actiunea cu scorul cel mai mare (argmax)
        return np.argmax(logits)

    def update(self, rollouts, sigma_rewards):

        # Sort rollouts by max(r_pos, r_neg) descending
        rollouts_sorted = sorted(rollouts,
                                  key=lambda x: max(x[0], x[1]),
                                  reverse=True)

        # Take only top-k best directions
        top_rollouts = rollouts_sorted[:self.num_best_deltas]

        step = np.zeros(self.weights.shape)
        for r_positive, r_negative, delta in top_rollouts:
            step += (r_positive - r_negative) * delta

        # Update weights
        self.weights += self.lr / (self.num_best_deltas * sigma_rewards + 1e-8) * step

    def decay_hyperparameters(self, current_iter, total_iters):

        progress = current_iter / total_iters

        # Linear decay to 30% of initial values
        decay_factor = 1.0 - 0.7 * progress

        self.lr = self.initial_lr * decay_factor
        self.noise = self.initial_noise * decay_factor

    def save(self, path):
        """Save agent weights and normalization stats"""
        np.savez(path,
                 weights=self.weights,
                 state_mean=self.state_mean,
                 state_std=self.state_std,
                 state_n=self.state_n)

    def load(self, path):
        """Load agent weights and normalization stats"""
        # Handle both .npy and .npz files
        if path.endswith('.npy'):
            self.weights = np.load(path)
        else:
            data = np.load(path)
            self.weights = data['weights']
            if 'state_mean' in data:
                self.state_mean = data['state_mean']
                self.state_std = data['state_std']
                self.state_n = int(data['state_n'])

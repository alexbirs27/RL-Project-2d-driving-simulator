import numpy as np


class RunningMeanStd:

    def __init__(self, shape):

        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # Small value to avoid division by zero

    def update(self, x):
        
        if x.ndim == 1:
            x = x.reshape(1, -1)

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        # Combine batch statistics with running statistics
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x):

        # Normalize input using running statistics.
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class ARSAgent:
    
    def __init__(self, state_dim, action_dim, learning_rate=0.02, noise=0.03,
                 num_deltas=16, top_k=None, total_iterations=500,
                 normalize_states=True):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_deltas = num_deltas
        self.top_k = top_k if top_k is not None else num_deltas
        self.total_iterations = total_iterations
        self.normalize_states = normalize_states

        # Initial hyperparameters 
        self.initial_lr = learning_rate
        self.initial_noise = noise
        self.lr = learning_rate
        self.noise = noise

        # Track current iteration for decay scheduling
        self.current_iteration = 0

        # Linear policy: Weight matrix that maps observations to action scores
        self.weights = np.zeros((action_dim, state_dim))

        # Running state normalization
        if normalize_states:
            self.state_normalizer = RunningMeanStd(state_dim)
        else:
            self.state_normalizer = None

    def _update_hyperparameters(self):
        
        progress = min(self.current_iteration / self.total_iterations, 1.0)
        # Linear decay to 10% of initial value
        self.lr = self.initial_lr * (1.0 - 0.9 * progress)
        self.noise = self.initial_noise * (1.0 - 0.9 * progress)

    def normalize_state(self, state, update_stats=False):
        # Normalize a state using running statistics.

        if self.state_normalizer is None:
            return state

        if update_stats:
            self.state_normalizer.update(state)

        return self.state_normalizer.normalize(state)

    def select_action(self, state, delta=None, direction=None, update_stats=False):
        
        # Normalize state
        normalized_state = self.normalize_state(state, update_stats=update_stats)

        # Start with a copy of current weights
        weights = self.weights.copy()

        # If training, apply perturbation in the specified direction
        if delta is not None:
            if direction == "plus":
                weights += delta * self.noise  # weights + noise
            else:
                weights -= delta * self.noise  # weights - noise

        # Result is a vector of scores for each action
        logits = weights.dot(normalized_state)

        # Return the action with the highest score
        return np.argmax(logits)

    def update(self, rollouts, sigma_rewards):
        
        # Update weights based on collected rollouts using top-k selection.

        # Update hyperparameters with decay
        self._update_hyperparameters()
        self.current_iteration += 1

        # Sort rollouts by max reward to select top-k
        rollouts_sorted = sorted(rollouts, key=lambda x: max(x[0], x[1]), reverse=True)
        top_rollouts = rollouts_sorted[:self.top_k]

        # Accumulate the weighted sum of deltas from top performers
        step = np.zeros(self.weights.shape)

        for r_positive, r_negative, delta in top_rollouts:
            # r_positive > r_negative: move toward +delta direction
            # r_negative > r_positive: move toward -delta direction 
            step += (r_positive - r_negative) * delta

        # Update weights with normalized step
        if sigma_rewards > 0:
            self.weights += self.lr / (self.top_k * sigma_rewards) * step
        else:
            self.weights += self.lr / self.top_k * step

    def save(self, path):

        save_dict = {
            'weights': self.weights,
            'current_iteration': np.array([self.current_iteration]),
            'lr': np.array([self.lr]),
            'noise': np.array([self.noise])
        }

        # Save normalization statistics if enabled
        if self.state_normalizer is not None:
            save_dict['norm_mean'] = self.state_normalizer.mean
            save_dict['norm_var'] = self.state_normalizer.var
            save_dict['norm_count'] = np.array([self.state_normalizer.count])

        np.savez(path, **save_dict)

    def load(self, path):
        
        # Handle both .npy (old format) and .npz (new format)
        if path.endswith('.npy'):
            # Legacy format: just weights
            self.weights = np.load(path)
        else:
            checkpoint = np.load(path)
            self.weights = checkpoint['weights']

            # Restore training state if available
            if 'current_iteration' in checkpoint:
                self.current_iteration = int(checkpoint['current_iteration'][0])
            if 'lr' in checkpoint:
                self.lr = float(checkpoint['lr'][0])
            if 'noise' in checkpoint:
                self.noise = float(checkpoint['noise'][0])

            # Restore normalization statistics if available
            if self.state_normalizer is not None and 'norm_mean' in checkpoint:
                self.state_normalizer.mean = checkpoint['norm_mean']
                self.state_normalizer.var = checkpoint['norm_var']
                self.state_normalizer.count = float(checkpoint['norm_count'][0])

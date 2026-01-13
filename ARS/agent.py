"""
ARS (Augmented Random Search) Agent

A simple, gradient-free reinforcement learning algorithm that uses random
perturbations to find good policy parameters. Works by:
1. Adding random noise to weights and evaluating performance
2. Comparing positive vs negative perturbations
3. Moving weights toward directions that improve reward

Key advantages:
- No neural networks needed (just a weight matrix)
- Very fast training (no backpropagation)
- Works well for simple control tasks
- Easy to understand and debug

Reference: Mania et al. (2018) - "Simple random search provides a
competitive approach to reinforcement learning"
"""

import numpy as np


class ARSAgent:
    """
    Linear policy agent trained with Augmented Random Search.

    The policy is a simple linear function: action = argmax(W @ state)
    where W is a weight matrix of shape (action_dim, state_dim).
    """

    def __init__(self, state_dim, action_dim, learning_rate=0.02, noise=0.03, num_deltas=16):
        """
        Initialize the ARS agent.

        Args:
            state_dim: Number of features in the observation (e.g., 10)
            action_dim: Number of possible actions (e.g., 9)
            learning_rate: Step size for weight updates (higher = faster but less stable)
            noise: Scale of random perturbations (higher = more exploration)
            num_deltas: Number of random directions to try per iteration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.noise = noise
        self.num_deltas = num_deltas

        # Linear policy: Weight matrix that maps observations to action scores
        # Shape: (action_dim, state_dim) = (9, 10) for our racing task
        self.weights = np.zeros((action_dim, state_dim))

    def select_action(self, state, delta=None, direction=None):
        """
        Select an action given the current state.

        During training, we add noise (delta) to explore different policies.
        During evaluation, we use the learned weights directly.

        Args:
            state: Current observation array of shape (state_dim,)
            delta: Random perturbation matrix (only used during training)
            direction: "plus" or "minus" - which direction to perturb

        Returns:
            action: Integer index of the selected action (0-8)
        """
        # Start with a copy of current weights
        weights = self.weights.copy()

        # If training, apply perturbation in the specified direction
        if delta is not None:
            if direction == "plus":
                weights += delta * self.noise  # Try weights + noise
            else:
                weights -= delta * self.noise  # Try weights - noise

        # Compute action scores: matrix multiplication (action_dim, state_dim) @ (state_dim,)
        # Result is a vector of scores for each action
        logits = weights.dot(state)

        # Return the action with the highest score
        return np.argmax(logits)

    def update(self, rollouts, sigma_rewards):
        """
        Update weights based on collected rollouts.

        The key insight: if a perturbation in direction +delta gave higher reward
        than -delta, we should move our weights in the +delta direction.

        Args:
            rollouts: List of (reward_plus, reward_minus, delta) tuples
                - reward_plus: Reward from policy (weights + noise*delta)
                - reward_minus: Reward from policy (weights - noise*delta)
                - delta: The random perturbation that was used
            sigma_rewards: Standard deviation of all rewards (for normalization)
        """
        # Accumulate the weighted sum of deltas
        step = np.zeros(self.weights.shape)

        for r_positive, r_negative, delta in rollouts:
            # If r_positive > r_negative: move toward +delta direction
            # If r_negative > r_positive: move toward -delta direction (negative contribution)
            step += (r_positive - r_negative) * delta

        # Update weights with normalized step
        # Dividing by (num_deltas * sigma_rewards) normalizes the update magnitude
        self.weights += self.lr / (self.num_deltas * sigma_rewards) * step

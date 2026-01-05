"""Training configuration for all agents."""

# PPO Configuration
PPO_CONFIG = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'total_timesteps': 100000  # ~50 episodes with max_steps=2000
}

# A2C Configuration
A2C_CONFIG = {
    'learning_rate': 7e-4,
    'n_steps': 5,
    'gamma': 0.99,
    'gae_lambda': 1.0,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'total_timesteps': 100000
}

# ARS Configuration
ARS_CONFIG = {
    'learning_rate': 0.02,
    'noise_std': 0.03,
    'num_deltas': 16,
    'num_best_deltas': 8,
    'num_iterations': 200,  # Number of training iterations
    'num_rollouts_per_delta': 3  # Rollouts per direction
}

# Environment Configuration
ENV_CONFIG = {
    'max_steps': 2000,
    'num_static_obstacles': 8,
    'num_moving_obstacles': 4
}

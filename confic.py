"""
Configuration file for different training experiments.
"""

# Base configuration
BASE_CONFIG = {
    'num_episodes': 5000,
    'max_steps_per_episode': 3000,
    'update_frequency': 10,
    'save_frequency': 100,
    'render_frequency': 100,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'save_dir': 'models'
}

# Quick test configuration (for debugging)
QUICK_CONFIG = {
    'num_episodes': 50,
    'max_steps_per_episode': 1000,
    'update_frequency': 5,
    'save_frequency': 25,
    'render_frequency': 10,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'save_dir': 'quick_models'
}

# Intensive training configuration
INTENSIVE_CONFIG = {
    'num_episodes': 10000,
    'max_steps_per_episode': 5000,
    'update_frequency': 20,
    'save_frequency': 200,
    'render_frequency': 500,
    'learning_rate': 1e-4,
    'gamma': 0.995,
    'save_dir': 'intensive_models'
}

# Fast convergence configuration (aggressive learning)
FAST_CONFIG = {
    'num_episodes': 2000,
    'max_steps_per_episode': 2000,
    'update_frequency': 5,
    'save_frequency': 50,
    'render_frequency': 50,
    'learning_rate': 5e-4,
    'gamma': 0.98,
    'save_dir': 'fast_models'
}

# Stable learning configuration (conservative)
STABLE_CONFIG = {
    'num_episodes': 8000,
    'max_steps_per_episode': 4000,
    'update_frequency': 15,
    'save_frequency': 150,
    'render_frequency': 200,
    'learning_rate': 1e-4,
    'gamma': 0.995,
    'save_dir': 'stable_models'
}


def get_config(name: str = 'base') -> dict:
    """
    Get configuration by name.
    
    Args:
        name: Configuration name ('base', 'quick', 'intensive', 'fast', 'stable')
    
    Returns:
        Configuration dictionary
    """
    configs = {
        'base': BASE_CONFIG,
        'quick': QUICK_CONFIG,
        'intensive': INTENSIVE_CONFIG,
        'fast': FAST_CONFIG,
        'stable': STABLE_CONFIG
    }
    
    if name not in configs:
        print(f"Warning: Unknown config '{name}', using 'base'")
        return BASE_CONFIG
    
    return configs[name]


def print_config(config: dict):
    """Print configuration in a readable format."""
    print("=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key:30s}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    # Display all configurations
    configs = ['base', 'quick', 'intensive', 'fast', 'stable']
    
    for config_name in configs:
        config = get_config(config_name)
        print(f"\n{config_name.upper()} Configuration:")
        print_config(config)
import yaml
import json
from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    """Base training configuration."""
    num_episodes: int = 5000
    max_steps_per_episode: int = 3000
    save_frequency: int = 100
    render_frequency: int = 50
    eval_frequency: int = 100
    gamma: float = 0.99
    device: str = 'auto'


@dataclass
class A2CConfig(TrainingConfig):
    """A2C-specific configuration."""
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_frequency: int = 10
    hidden_dim: int = 256


@dataclass
class PPOConfig(TrainingConfig):
    """PPO-specific configuration."""
    learning_rate: float = 3e-4
    lam: float = 0.95
    clip_eps: float = 0.2
    steps_per_epoch: int = 4096
    train_iters: int = 10
    minibatch_size: int = 64
    hidden_dim: int = 256


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")


def save_config(config: Any, config_path: str):
    """Save configuration to YAML or JSON file."""
    if hasattr(config, '__dict__'):
        config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__
    else:
        config_dict = config

    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    elif config_path.endswith('.json'):
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")


def create_default_configs():
    """Create default configuration files."""
    import os

    # Create configs directory
    os.makedirs('configs', exist_ok=True)

    # A2C config
    a2c_config = A2CConfig()
    save_config(a2c_config, 'configs/a2c_config.yaml')
    print("Created configs/a2c_config.yaml")

    # PPO config
    ppo_config = PPOConfig()
    save_config(ppo_config, 'configs/ppo_config.yaml')
    print("Created configs/ppo_config.yaml")


if __name__ == '__main__':
    create_default_configs()

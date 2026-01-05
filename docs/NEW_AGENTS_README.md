# 2D Driving Simulator with Reinforcement Learning

A comprehensive 2D top-down racing game with multiple reinforcement learning agents. Train AI to learn autonomous driving using state-of-the-art RL algorithms: **ARS**, **A2C**, and **PPO**.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Multiple RL Algorithms**: ARS (evolution-based), A2C (actor-critic), PPO (state-of-the-art)
- **Realistic Physics**: Velocity, acceleration, friction, off-road penalties
- **Advanced Sensors**: 8-directional raycasting for environment perception
- **Comprehensive Logging**: CSV logs, training plots, model checkpoints
- **Headless Training**: Fast training without visualization
- **Interactive Evaluation**: Watch trained agents drive in real-time
- **Comparison Tools**: Benchmark and compare multiple agents

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd RL-Project-2d-driving-simulator

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Gymnasium
- Pygame
- NumPy
- Matplotlib

### Train Your First Agent

```bash
# Interactive menu (recommended for beginners)
python src/training/train_all_agents.py

# Or train specific agents directly:
python src/training/train_ars.py      # ARS (fastest, 10-20 min)
python src/training/train_a2c.py      # A2C (2-4 hours)
python src/agents/PPO/train.py        # PPO (3-6 hours, best performance)
```

### Play Manually

```bash
python src/game/play_manual.py
```

**Controls**:
- **Arrow Keys**: Drive (UP=accelerate, DOWN=brake, LEFT/RIGHT=turn)
- **R**: Restart
- **ESC**: Quit

## Project Structure

```
RL-Project-2d-driving-simulator/
├── src/
│   ├── agents/              # RL agent implementations
│   │   ├── ars/            # Augmented Random Search
│   │   ├── a2c/            # Advantage Actor-Critic
│   │   ├── PPO/            # Proximal Policy Optimization
│   │   └── random/         # Random baseline
│   ├── env/                # Gymnasium environments
│   │   ├── driving_env.py  # Main training environment
│   │   └── racing_env.py   # Alternative environment
│   ├── game/               # Game engine components
│   │   ├── engine.py       # Core game loop
│   │   ├── car.py          # Vehicle physics
│   │   ├── track.py        # Track generation & collision
│   │   ├── renderer.py     # Pygame visualization
│   │   └── play_manual.py  # Manual control
│   ├── training/           # Training scripts
│   │   ├── train_ars.py
│   │   ├── train_a2c.py
│   │   ├── train_all_agents.py  # Interactive menu
│   │   └── compare_agents.py    # Multi-agent comparison
│   ├── utils/              # Utilities
│   │   └── logger.py       # Advanced logging
│   └── config.py           # Global configuration
├── configs/                # Algorithm configurations
│   ├── training_config.py
│   ├── a2c_config.yaml
│   └── ppo_config.yaml
├── docs/                   # Documentation
│   └── AGENTS.md          # Comprehensive agent documentation
├── models/                 # Saved model weights
├── logs/                   # Training logs (CSV)
├── plots/                  # Training curves
└── README.md
```

## Agents Overview

### 1. ARS (Augmented Random Search)

**Type**: Evolution-based
**Training Time**: 10-20 minutes
**Parameters**: 65 (linear policy)

**Pros**:
- Extremely fast to train
- No neural networks
- Very simple implementation
- Low computational requirements

**Cons**:
- Limited expressiveness
- Plateaus quickly
- No memory/history

**Best For**: Quick prototyping, baselines, resource-constrained environments

```bash
python src/training/train_ars.py
```

### 2. A2C (Advantage Actor-Critic)

**Type**: Policy gradient with value function
**Training Time**: 2-4 hours
**Parameters**: ~37,000

**Pros**:
- Good balance of speed/performance
- Stable training
- Moderate sample efficiency
- Entropy regularization for exploration

**Cons**:
- Sensitive to hyperparameters
- On-policy (can't reuse data)
- Slower than evolution methods

**Best For**: Balanced performance, moderate computational budgets

```bash
python src/training/train_a2c.py
```

### 3. PPO (Proximal Policy Optimization)

**Type**: Clipped policy gradient
**Training Time**: 3-6 hours
**Parameters**: ~8,200

**Pros**:
- State-of-the-art performance
- Very stable training
- Robust to hyperparameters
- Industry standard (used in ChatGPT)

**Cons**:
- Slowest to train
- More complex implementation
- Higher computational cost

**Best For**: Maximum performance, production systems, when stability matters

```bash
python src/agents/PPO/train.py
```

## Environment Details

### Observation Space (13 dimensions)

- **Velocity** (normalized): Current speed
- **Angle** (sin/cos): Car orientation
- **8 Distance Sensors**: Raycasting in 8 directions (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
- **Progress**: Track completion (0-1)
- **On Road**: Boolean flag

### Action Space (5 discrete actions)

| Action | ID | Effect |
|--------|-----|--------|
| NONE | 0 | Coast |
| ACCELERATE | 1 | Speed up |
| BRAKE | 2 | Slow down |
| TURN_LEFT | 3 | Rotate CCW |
| TURN_RIGHT | 4 | Rotate CW |

### Reward Function

```python
reward = velocity_reward + progress_reward + penalties

where:
- velocity_reward = (velocity / max_velocity) × 1.0
- progress_reward = delta_progress × 1000
- collision_penalty = -2.0 (if off-road)
- stationary_penalty = -2.0 (if stuck)
- lap_completion_bonus = +5000
```

## Training & Evaluation

### Interactive Training Menu

```bash
python src/training/train_all_agents.py
```

**Options**:
1. **Test Agent with Visualization** - Load and watch a trained agent
2. **Train All Agents** - Sequential training of ARS, A2C, PPO
3. **Compare Models** - Generate performance comparison plots

### Training Individual Agents

Each training script:
- Creates unique timestamped runs
- Saves best model (highest reward)
- Saves final model
- Generates CSV logs
- Creates training plots automatically

**ARS**:
```bash
python src/training/train_ars.py
# Output: models/ars_run_<timestamp>_BEST_weights.npy
```

**A2C**:
```bash
python src/training/train_a2c.py
# Output: models/a2c_run_<timestamp>_BEST_model.pt
```

**PPO**:
```bash
python src/agents/PPO/train.py
# Output: models/ppo_run_<timestamp>_BEST_model.pt

# With visualization (slower):
python src/agents/PPO/train.py --render
```

### Comparing Agents

```bash
python src/training/compare_agents.py
```

Generates comparative plots and statistics for all trained models.

## Configuration

### Global Settings

Edit [src/config.py](src/config.py):

```python
# Environment
STATE_DIM = 13
ACTION_DIM = 5

# ARS
ARS_ITERATIONS = 200
ARS_NUM_DELTAS = 16
LR_ARS = 0.02

# A2C
A2C_EPISODES = 3000
LR_A2C = 1e-3

# Training
MAX_STEPS_PER_EPISODE = 3000
```

### Agent-Specific Configs

- **A2C**: [configs/a2c_config.yaml](configs/a2c_config.yaml)
- **PPO**: [configs/ppo_config.yaml](configs/ppo_config.yaml)
- **Training Profiles**: [confic.py](confic.py) (BASE, QUICK, INTENSIVE, etc.)

## Performance Benchmarks

Typical results after full training:

| Agent | Avg Reward | Best Reward | Success Rate | Training Time |
|-------|------------|-------------|--------------|---------------|
| **PPO** | 800-1500 | ~5000 | 15-30% | 3-6 hours |
| **A2C** | 500-1200 | ~4000 | 10-20% | 2-4 hours |
| **ARS** | 300-800 | ~2000 | 5-10% | 10-20 min |
| **Random** | -20 to +50 | ~100 | 0% | N/A |

**Success Rate** = % of episodes where car completes a full lap

## Documentation

For detailed information about each agent (theory, architecture, training, usage):

**[Read the Full Agent Documentation →](docs/AGENTS.md)**

Contents:
- Theoretical background for each algorithm
- Neural network architectures
- Training procedures and hyperparameters
- Usage examples and code snippets
- Troubleshooting and debugging tips
- Advanced usage and customization
- Performance comparisons

## Advanced Usage

### Custom Reward Function

Edit [src/env/driving_env.py](src/env/driving_env.py):

```python
def _calculate_reward(self):
    reward = 0.0

    # Your custom logic here
    reward += (self.velocity / self.max_velocity) * 1.0
    reward += delta_progress * 1000

    if self.collision:
        reward -= 2.0

    return reward
```

### Load and Evaluate Model

```python
from src.agents.ppo.agent import PPOAgent
from src.env.driving_env import make_env

# Load agent
agent = PPOAgent(state_dim=13, action_dim=5)
agent.load('models/ppo_run_2026-01-05_22-53-56_BEST_model.pt')

# Evaluate
env = make_env(render_mode="human")
state, _ = env.reset()

for _ in range(3000):
    action, _, _ = agent.act(state)
    state, reward, terminated, truncated, _ = env.step(action)
    env.render()

    if terminated or truncated:
        break

env.close()
```

### Hyperparameter Tuning

```python
# Example: Grid search for ARS learning rate
learning_rates = [0.01, 0.02, 0.05, 0.1]

for lr in learning_rates:
    agent = ARSAgent(learning_rate=lr)
    train(agent, run_name=f'ars_lr_{lr}')
```

## Troubleshooting

**Agent not learning?**
- Check reward function returns non-zero values
- Verify environment resets properly
- Try increasing learning rate
- Ensure observations are normalized

**Training unstable?**
- Reduce learning rate
- Add gradient clipping
- Increase batch size (PPO)
- Check for bugs in update logic

**Out of memory?**
- Reduce batch size
- Train on CPU
- Use gradient accumulation

**Import errors?**
- Ensure project root is in Python path
- Check `__init__.py` files exist
- Verify relative imports

## Future Enhancements

- [ ] Additional algorithms (SAC, TD3, DQN)
- [ ] Multiple track layouts
- [ ] Multi-agent racing
- [ ] Continuous action space
- [ ] TensorBoard integration
- [ ] Distributed training
- [ ] Automatic hyperparameter tuning

## Contributing

Contributions welcome! To add a new agent:

1. Create directory: `src/agents/<agent_name>/`
2. Implement standard interface (see existing agents)
3. Create training script: `src/training/train_<agent_name>.py`
4. Update documentation
5. Submit PR

## References

- **ARS**: Mania et al. (2018) - "Simple random search provides a competitive approach to reinforcement learning"
- **A2C**: Mnih et al. (2016) - "Asynchronous methods for deep reinforcement learning"
- **PPO**: Schulman et al. (2017) - "Proximal policy optimization algorithms"

## Resources

- [OpenAI Spinning Up](https://spinningup.openai.com/) - RL educational resource
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - Production RL implementations
- [Gymnasium Docs](https://gymnasium.farama.org/) - Environment API reference

## License

MIT License - See LICENSE file for details

## Acknowledgments

Developed as part of a Reinforcement Learning course. Thanks to:
- OpenAI Spinning Up team for educational materials
- Gymnasium developers for standardized APIs
- RL research community for algorithm innovations

---

**Version**: 1.0
**Last Updated**: 2026-01-05

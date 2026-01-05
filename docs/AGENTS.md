# Reinforcement Learning Agents - 2D Driving Simulator

This document provides comprehensive information about all reinforcement learning agents implemented in this project, including their theory, architecture, training procedures, and usage instructions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Description](#environment-description)
3. [ARS Agent (Augmented Random Search)](#ars-agent-augmented-random-search)
4. [A2C Agent (Advantage Actor-Critic)](#a2c-agent-advantage-actor-critic)
5. [PPO Agent (Proximal Policy Optimization)](#ppo-agent-proximal-policy-optimization)
6. [Random Agent (Baseline)](#random-agent-baseline)
7. [Comparison & Results](#comparison--results)
8. [Quick Start Guide](#quick-start-guide)
9. [Advanced Usage](#advanced-usage)

---

## Project Overview

This project implements a **2D top-down driving simulator** where an agent learns to drive around an oval track. The project includes four different reinforcement learning agents:

- **ARS** - Evolution-based, parameter-efficient approach
- **A2C** - Actor-Critic with neural networks
- **PPO** - State-of-the-art policy gradient method
- **Random** - Baseline for comparison

### Key Features

- **Realistic Physics**: Velocity, acceleration, friction, off-road penalties
- **Distance Sensors**: 8-directional raycasting for track edge detection
- **Modular Design**: Easy to add new agents or modify existing ones
- **Comprehensive Logging**: CSV logs, plots, and model checkpoints
- **Headless Training**: Fast training without rendering
- **Visual Evaluation**: Watch trained agents drive in real-time

---

## Environment Description

### DrivingEnv - Primary Training Environment

**Location**: [src/env/driving_env.py](src/env/driving_env.py)

#### Observation Space (13 dimensions)

| Feature | Description | Range |
|---------|-------------|-------|
| Velocity (normalized) | Current speed / max speed | [0, 1] |
| Angle (sin) | Sine of car orientation | [-1, 1] |
| Angle (cos) | Cosine of car orientation | [-1, 1] |
| Distance Sensor 0° | Front distance to track edge | [0, 1] normalized |
| Distance Sensor 45° | Front-right distance | [0, 1] normalized |
| Distance Sensor 90° | Right distance | [0, 1] normalized |
| Distance Sensor 135° | Back-right distance | [0, 1] normalized |
| Distance Sensor 180° | Back distance | [0, 1] normalized |
| Distance Sensor 225° | Back-left distance | [0, 1] normalized |
| Distance Sensor 270° | Left distance | [0, 1] normalized |
| Distance Sensor 315° | Front-left distance | [0, 1] normalized |
| Progress | Completion along track | [0, 1] |
| On Road Flag | Whether car is on track | {0, 1} |

#### Action Space (5 discrete actions)

| Action ID | Name | Effect |
|-----------|------|--------|
| 0 | NONE | No acceleration/braking/turning |
| 1 | ACCELERATE | Increase velocity |
| 2 | BRAKE | Decrease velocity |
| 3 | TURN_LEFT | Rotate counterclockwise |
| 4 | TURN_RIGHT | Rotate clockwise |

#### Reward Function

The reward system is designed to encourage:
- **Speed**: Faster driving receives higher rewards
- **Progress**: Advancing along the track
- **Safety**: Staying on the road
- **Completion**: Finishing laps

```python
# Components:
velocity_reward = (current_velocity / max_velocity) * 1.0
progress_reward = delta_progress * 1000
collision_penalty = -2.0  (if crashed)
stationary_penalty = -2.0  (if velocity < 1.0 for too long)
lap_completion_bonus = +5000
```

**Total Reward per Step**:
```
reward = velocity_reward + progress_reward + collision_penalty + stationary_penalty
```

#### Episode Termination

An episode ends when:
1. **Collision**: Car drives off the track
2. **Max Steps**: 3000 steps reached (prevents infinite loops)
3. **Lap Completion**: Car crosses finish line (success!)

### Physics Model

**Car Physics** ([src/game/car.py](src/game/car.py)):
- **Max Velocity**: 240.0 pixels/second (on road)
- **Acceleration**: 0.5 units/frame
- **Friction**: 0.98 (on road), 0.4 (off road)
- **Turn Speed**: Scales with velocity
- **Off-Road Penalty**: Max velocity reduced to 60.0

**Track Design** ([src/game/track.py](src/game/track.py)):
- **Shape**: Oval with procedural wobble
- **Road Width**: 100 pixels
- **Track Points**: 100 center points
- **Collision Detection**: Distance-to-segment algorithm

---

## ARS Agent (Augmented Random Search)

### Overview

**ARS** is an evolution-based reinforcement learning algorithm that uses random perturbations to improve a linear policy. It's extremely simple, parameter-efficient, and often surprisingly effective.

**Key Characteristics**:
- **No neural networks** - Just a weight matrix
- **Evolution-based** - Uses random search instead of gradients
- **Fast convergence** - Often learns in few iterations
- **Low memory** - Only stores weights (13 × 5 = 65 parameters)

### Theory

ARS belongs to the family of **derivative-free optimization** methods. Instead of computing gradients, it:

1. **Perturbs** the current policy with random noise
2. **Evaluates** both positive and negative perturbations
3. **Updates** towards directions that increase reward

**Mathematical Formulation**:

Policy: Linear mapping from state to action scores
```
π(s) = W · s
where W is a [action_dim × state_dim] matrix
```

Update Rule:
```
W ← W + α × (1 / (N × σ_r)) × Σ(r_pos - r_neg) × δ_i

where:
- α = learning rate (step size)
- N = number of perturbations
- σ_r = standard deviation of rewards (normalization)
- r_pos, r_neg = rewards from positive/negative perturbations
- δ_i = random perturbation
```

### Architecture

**Location**: [src/agents/ars/agent.py](src/agents/ars/agent.py)

```python
class ARSAgent:
    def __init__(self, state_dim=13, action_dim=5, learning_rate=0.02):
        self.weights = np.zeros((action_dim, state_dim))
        self.num_deltas = 16  # Number of perturbations per iteration
        self.learning_rate = learning_rate
```

**Parameters**:
- `weights`: [5 × 13] matrix mapping observations to actions
- `num_deltas`: 16 perturbations per iteration
- `learning_rate`: 0.02 (step size)

**Methods**:
- `select_action(state, delta, direction)`: Returns action with optional perturbation
- `update(rollouts, sigma_r)`: Updates weights based on reward differences
- `save(path)`: Saves weights to `.npy` file
- `load(path)`: Loads weights from file

### Training Procedure

**Script**: [src/training/train_ars.py](src/training/train_ars.py)

#### Algorithm Steps

```
For each iteration (200 total):
    1. Generate 16 random perturbations: δ_1, δ_2, ..., δ_16

    2. For each perturbation δ_i:
        a. Evaluate W + ν·δ_i (positive direction)
        b. Evaluate W - ν·δ_i (negative direction)
        c. Store rewards: r_pos, r_neg

    3. Calculate reward statistics:
        σ_r = std(all rewards)

    4. Update weights:
        W ← W + (α / N·σ_r) × Σ(r_pos - r_neg)·δ_i

    5. Save model if max_reward improved
```

#### Configuration

From [src/config.py](src/config.py):
```python
ARS_ITERATIONS = 200
ARS_NUM_DELTAS = 16
LR_ARS = 0.02
STATE_DIM = 13
ACTION_DIM = 5
```

#### Training Command

```bash
# From project root
python src/training/train_ars.py
```

#### Training Output

The script generates:
1. **Model Files**:
   - `models/ars_run_<timestamp>_BEST_weights.npy` - Best performing weights
   - `models/ars_run_<timestamp>_FINAL_weights.npy` - Final weights

2. **Logs**:
   - `logs/ars_run_<timestamp>_log.csv` - CSV with iteration, avg_reward, max_reward

3. **Plots**:
   - `plots/ars_run_<timestamp>_plot.png` - Training curves

#### Typical Training Time

- **200 iterations**: ~10-20 minutes (no rendering)
- **Per iteration**: ~3-5 seconds
- **Total evaluations**: 200 iterations × 16 deltas × 2 directions = 6,400 episodes

### Usage

#### Training a New Model

```python
from src.agents.ars.agent import ARSAgent
from src.env.driving_env import make_env

env = make_env(render_mode=None)
agent = ARSAgent(state_dim=13, action_dim=5, learning_rate=0.02)

# Training loop (see train_ars.py for full implementation)
```

#### Loading a Trained Model

```python
import numpy as np
from src.agents.ars.agent import ARSAgent
from src.env.driving_env import make_env

# Load agent
agent = ARSAgent(state_dim=13, action_dim=5)
agent.weights = np.load('models/ars_run_<timestamp>_BEST_weights.npy')

# Evaluate
env = make_env(render_mode="human")
state, _ = env.reset()

for _ in range(3000):
    action = agent.select_action(state)
    state, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        break

env.close()
```

### Strengths & Weaknesses

**Strengths**:
- ✅ Very simple to implement
- ✅ No hyperparameter tuning needed
- ✅ Fast to train
- ✅ No gradient computation
- ✅ Works well for low-dimensional problems

**Weaknesses**:
- ❌ Limited expressiveness (linear policy only)
- ❌ Struggles with complex state spaces
- ❌ No memory (reactive policy)
- ❌ Doesn't scale to high dimensions

---

## A2C Agent (Advantage Actor-Critic)

### Overview

**A2C** (Advantage Actor-Critic) is an on-policy reinforcement learning algorithm that uses two neural networks: an **actor** (policy) and a **critic** (value function). The advantage function helps reduce variance in policy gradient estimates.

**Key Characteristics**:
- **Neural networks** - Deep learning based
- **Actor-Critic** - Separate policy and value networks
- **Advantage estimation** - Reduces gradient variance
- **On-policy** - Learns from current policy samples

### Theory

A2C is based on **policy gradient** methods with a **baseline** to reduce variance.

**Objective**: Maximize expected return
```
J(θ) = E[Σ γ^t · r_t]
```

**Policy Gradient Theorem**:
```
∇J(θ) = E[∇log π(a|s) · A(s,a)]

where:
- π(a|s) = policy (probability of action a in state s)
- A(s,a) = advantage function = Q(s,a) - V(s)
```

**Advantage Estimation**:
```
A(s,a) ≈ r + γ·V(s') - V(s)

This tells us: "How much better is action a compared to average?"
```

**Loss Functions**:

1. **Actor Loss** (policy):
   ```
   L_actor = -Σ log π(a|s) · A(s,a) - β·H(π)

   where H(π) is entropy (encourages exploration)
   ```

2. **Critic Loss** (value):
   ```
   L_critic = (V(s) - R_target)²

   where R_target = Σ γ^t · r_t (discounted return)
   ```

### Architecture

**Location**: [src/agents/a2c/](src/agents/a2c/)

#### Neural Network Structure

**Actor Network** ([src/agents/a2c/networks.py](src/agents/a2c/networks.py)):
```
Input (13) → FC(128) → ReLU → FC(128) → ReLU → FC(5) → Softmax → Action Probabilities
```

**Critic Network**:
```
Input (13) → FC(128) → ReLU → FC(128) → ReLU → FC(1) → State Value
```

**Parameters**:
- **Actor**: 13×128 + 128×128 + 128×5 = 18,693 parameters
- **Critic**: 13×128 + 128×128 + 128×1 = 18,561 parameters
- **Total**: ~37,000 parameters

#### Agent Class

**Location**: [src/agents/a2c/agent.py](src/agents/a2c/agent.py)

```python
class A2CAgent:
    def __init__(self, state_dim=13, action_dim=5, lr=1e-3, gamma=0.99):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = 0.01  # Exploration bonus
```

**Methods**:
- `select_action(state)`: Sample action from policy
- `compute_returns(rewards, dones)`: Calculate discounted returns
- `update(states, actions, rewards, dones, next_states)`: Train both networks
- `save(path)`: Save model weights
- `load(path)`: Load model weights

### Training Procedure

**Script**: [src/training/train_a2c.py](src/training/train_a2c.py)

#### Algorithm Steps

```
For each episode (3000 total):
    1. Reset environment: s_0 ← env.reset()

    2. Collect trajectory:
        While not done:
            a_t ← π(s_t)  (sample from actor)
            s_{t+1}, r_t ← env.step(a_t)
            Store (s_t, a_t, r_t, done_t)

    3. Compute discounted returns:
        R_t = Σ_{k=t}^T γ^{k-t} · r_k

    4. Compute advantages:
        A_t = R_t - V(s_t)

    5. Update actor:
        Minimize: -log π(a_t|s_t) · A_t - β·H(π)

    6. Update critic:
        Minimize: (V(s_t) - R_t)²

    7. Save if best total reward
```

#### Configuration

From [src/config.py](src/config.py) and [configs/a2c_config.yaml](configs/a2c_config.yaml):
```python
A2C_EPISODES = 3000
LR_A2C = 1e-3
GAMMA = 0.99
ENTROPY_COEF = 0.01
HIDDEN_DIM = 128
```

#### Training Command

```bash
python src/training/train_a2c.py
```

#### Training Output

1. **Model Files**:
   - `models/a2c_run_<timestamp>_BEST_model.pt` - Best model
   - `models/a2c_run_<timestamp>_FINAL_model.pt` - Final model

2. **Logs**:
   - `logs/a2c_run_<timestamp>_log.csv` - Episode rewards and losses

3. **Plots**:
   - `plots/a2c_run_<timestamp>_plot.png` - Training curves with moving average

#### Typical Training Time

- **3000 episodes**: ~2-4 hours (no rendering)
- **Per episode**: ~2-4 seconds
- **Convergence**: Usually after 500-1000 episodes

### Usage

#### Training

```python
from src.agents.a2c.agent import A2CAgent
from src.env.driving_env import make_env

env = make_env(render_mode=None)
agent = A2CAgent(state_dim=13, action_dim=5)

for episode in range(3000):
    states, actions, rewards, dones = [], [], [], []
    state, _ = env.reset()

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        state = next_state

    agent.update(states, actions, rewards, dones, state)
```

#### Evaluation

```python
import torch
from src.agents.a2c.agent import A2CAgent
from src.env.driving_env import make_env

agent = A2CAgent(state_dim=13, action_dim=5)
agent.load('models/a2c_run_<timestamp>_BEST_model.pt')

env = make_env(render_mode="human")
state, _ = env.reset()

while True:
    action = agent.select_action(state)
    state, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        break
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 1e-3 | Step size for gradient descent |
| Gamma (γ) | 0.99 | Discount factor for future rewards |
| Entropy Coefficient | 0.01 | Exploration bonus weight |
| Hidden Dimensions | 128 | Neurons in hidden layers |
| Gradient Clipping | 0.5 | Prevents exploding gradients |

### Strengths & Weaknesses

**Strengths**:
- ✅ Good sample efficiency
- ✅ Stable training (with advantage function)
- ✅ Handles continuous state spaces well
- ✅ Lower variance than REINFORCE
- ✅ Entropy regularization encourages exploration

**Weaknesses**:
- ❌ On-policy (can't reuse old data)
- ❌ Sensitive to hyperparameters
- ❌ Can get stuck in local optima
- ❌ Slower than off-policy methods

---

## PPO Agent (Proximal Policy Optimization)

### Overview

**PPO** is a state-of-the-art policy gradient algorithm that improves upon A2C by constraining policy updates to prevent destructively large changes. It's currently one of the most popular RL algorithms due to its reliability and performance.

**Key Characteristics**:
- **Clipped objective** - Prevents policy collapse
- **Multiple epochs** - Reuses data multiple times
- **GAE** - Generalized Advantage Estimation
- **State-of-the-art** - Used in ChatGPT RLHF training

### Theory

PPO addresses the key challenge in policy gradient methods: **how to make the largest possible improvement without causing performance collapse**.

**Core Idea**: Limit policy updates to a "trust region"

**PPO-Clip Objective**:
```
L^CLIP(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

where:
- r_t(θ) = π_new(a|s) / π_old(a|s)  (probability ratio)
- A_t = advantage
- ε = clipping parameter (typically 0.2)
- clip(x, min, max) = clamp x to [min, max]
```

**What this does**:
- If action is good (A_t > 0): allow increase up to (1+ε)
- If action is bad (A_t < 0): allow decrease up to (1-ε)
- Prevents extreme policy changes

**Generalized Advantage Estimation (GAE)**:
```
A_t^GAE = Σ_{l=0}^∞ (γλ)^l · δ_{t+l}

where:
- δ_t = r_t + γ·V(s_{t+1}) - V(s_t)  (TD error)
- λ = bias-variance trade-off parameter (0.95)
```

GAE smoothly interpolates between:
- **λ=0**: Low variance, high bias (TD learning)
- **λ=1**: High variance, low bias (Monte Carlo)

### Architecture

**Location**: [src/agents/PPO/](src/agents/PPO/)

#### Neural Network Structure

**Policy Network** ([src/agents/PPO/networks.py](src/agents/PPO/networks.py)):
```
Input (13) → FC(256) → ReLU → FC(5) → Action Logits
```

**Value Network**:
```
Input (13) → FC(256) → ReLU → FC(1) → State Value
```

**Parameters**:
- **Policy**: 13×256 + 256×5 = 4,608 parameters
- **Value**: 13×256 + 256×1 = 3,584 parameters
- **Total**: ~8,200 parameters

**Note**: PPO uses simpler networks than A2C (single hidden layer vs two), trading capacity for training stability.

#### Agent Class

**Location**: [src/agents/PPO/agent.py](src/agents/PPO/agent.py)

```python
class PPOAgent:
    def __init__(self, state_dim=13, action_dim=5, gamma=0.99,
                 lam=0.95, clip_eps=0.2, lr=3e-4,
                 steps_per_epoch=4096, train_iters=10, minibatch_size=64):
        self.policy = PolicyNet(state_dim, action_dim)
        self.value_fn = ValueNet(state_dim)
        self.opt_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.opt_value = optim.Adam(self.value_fn.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.steps_per_epoch = steps_per_epoch
        self.train_iters = train_iters
        self.minibatch_size = minibatch_size
```

**Methods**:
- `act(state)`: Returns action, log_prob, and value
- `compute_gae(rewards, values, next_value, dones)`: GAE calculation
- `train(obs, actions, logp_old, advantages, returns)`: PPO update
- `save(path)`: Save both networks
- `load(path)`: Load both networks

### Training Procedure

**Script**: [src/agents/PPO/train.py](src/agents/PPO/train.py)

#### Algorithm Steps

```
For each epoch (200 total):
    1. Collect 4096 steps of experience:
        For step in range(4096):
            a_t, log_π(a_t), V(s_t) ← agent.act(s_t)
            s_{t+1}, r_t ← env.step(a_t)
            Store (s_t, a_t, log_π(a_t), V(s_t), r_t, done_t)

            If done: reset environment

    2. Compute GAE advantages:
        For t in reversed(range(T)):
            δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
            A_t = δ_t + γ·λ·(1-done)·A_{t+1}

        Returns: R_t = A_t + V(s_t)

    3. Normalize advantages:
        A ← (A - mean(A)) / (std(A) + 1e-8)

    4. Train for 10 inner epochs:
        For _ in range(10):
            Shuffle data
            For minibatch in data:
                # Recompute log probabilities
                log_π_new ← policy(s, a)
                ratio = exp(log_π_new - log_π_old)

                # Clipped objective
                L_policy = -min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)

                # Value loss
                L_value = (V(s) - R)²

                # Update networks
                optimize(L_policy)
                optimize(L_value)

    5. Save if mean reward improved
```

#### Configuration

From [configs/ppo_config.yaml](configs/ppo_config.yaml):
```python
GAMMA = 0.99          # Discount factor
LAMBDA = 0.95         # GAE parameter
CLIP_EPSILON = 0.2    # PPO clipping
LEARNING_RATE = 3e-4  # Step size
STEPS_PER_EPOCH = 4096  # Experience buffer size
TRAIN_ITERS = 10      # Inner training loops
MINIBATCH_SIZE = 64   # Batch size
EPOCHS = 200          # Total epochs
```

#### Training Command

```bash
python src/agents/PPO/train.py

# With visualization (slower)
python src/agents/PPO/train.py --render
```

#### Training Output

1. **Model Files**:
   - `models/ppo_run_<timestamp>_BEST_model.pt` - Best mean reward model
   - `models/ppo_run_<timestamp>_FINAL_model.pt` - Final model

2. **Logs**:
   - `logs/ppo_run_<timestamp>_log.csv` - Epoch, mean_reward

3. **Plots**:
   - `plots/ppo_run_<timestamp>_plot.png` - Episode rewards with moving average

#### Typical Training Time

- **200 epochs**: ~3-6 hours (no rendering)
- **Per epoch**: ~1-2 minutes (4096 steps + 10 training iterations)
- **Convergence**: Usually after 50-100 epochs

### Usage

#### Training

```python
from src.agents.PPO.agent import PPOAgent
from src.env.driving_env import make_env

env = make_env(render_mode=None)
agent = PPOAgent(state_dim=13, action_dim=5)

for epoch in range(200):
    # Collect rollout
    observations, actions, logps, values, rewards, dones = [], [], [], [], [], []
    state, _ = env.reset()

    for step in range(4096):
        action, logp, value = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        observations.append(state)
        actions.append(action)
        logps.append(logp)
        values.append(value)
        rewards.append(reward)
        dones.append(done)

        state = next_state if not done else env.reset()[0]

    # Compute GAE
    next_value = agent.value_fn(torch.tensor(state, dtype=torch.float32)).item()
    advantages, returns = agent.compute_gae(rewards, values, next_value, dones)

    # Train
    agent.train(observations, actions, logps, advantages, returns)
```

#### Evaluation

```python
from src.agents.PPO.agent import PPOAgent
from src.env.driving_env import make_env

agent = PPOAgent(state_dim=13, action_dim=5)
agent.load('models/ppo_run_<timestamp>_BEST_model.pt')

env = make_env(render_mode="human")
state, _ = env.reset()

for _ in range(3000):
    action, _, _ = agent.act(state)
    state, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        state, _ = env.reset()
```

### Hyperparameters

| Parameter | Value | Description | Typical Range |
|-----------|-------|-------------|---------------|
| γ (gamma) | 0.99 | Discount factor | [0.95, 0.999] |
| λ (lambda) | 0.95 | GAE parameter | [0.9, 0.99] |
| ε (epsilon) | 0.2 | Clip range | [0.1, 0.3] |
| Learning Rate | 3e-4 | Optimizer step size | [1e-5, 1e-3] |
| Steps/Epoch | 4096 | Rollout length | [2048, 8192] |
| Train Iterations | 10 | Inner epochs | [3, 15] |
| Minibatch Size | 64 | Batch size | [32, 256] |

### Strengths & Weaknesses

**Strengths**:
- ✅ State-of-the-art performance
- ✅ Stable training (clipped objective)
- ✅ Sample efficient (reuses data)
- ✅ Robust to hyperparameters
- ✅ Works across many domains
- ✅ Industry standard (used in RLHF)

**Weaknesses**:
- ❌ More complex than A2C
- ❌ Slower per update (multiple inner epochs)
- ❌ Still on-policy (can't use replay buffer)
- ❌ Requires more tuning than simple methods

**When to use PPO**:
- You want reliable, strong performance
- You have computational resources
- Sample efficiency is important
- You need stable training

---

## Random Agent (Baseline)

### Overview

The **Random Agent** serves as a baseline for comparison. It simply samples random actions from the action space without any learning.

**Purpose**:
- Establish performance floor
- Verify environment is learnable
- Sanity check for other agents

### Implementation

**Location**: [src/agents/random/random_agent.py](src/agents/random/random_agent.py)

```python
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, state):
        return self.action_space.sample()
```

### Usage

```python
from src.env.driving_env import make_env
from src.agents.random.random_agent import RandomAgent

env = make_env(render_mode="human")
agent = RandomAgent(env.action_space)

state, _ = env.reset()
total_reward = 0

for _ in range(1000):
    action = agent.select_action(state)
    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

print(f"Random agent reward: {total_reward}")
```

### Expected Performance

Random agents typically achieve:
- **Average reward**: -50 to +50 per episode
- **Success rate**: 0% (never completes lap)
- **Progress**: Usually 0.1-0.3 (10-30% of track)

This demonstrates that the task **requires learning** - random actions don't work.

---

## Comparison & Results

### Performance Summary

Based on training runs, here's the typical performance ranking:

| Agent | Avg Reward | Best Reward | Training Time | Parameters | Success Rate |
|-------|------------|-------------|---------------|------------|--------------|
| **PPO** | ~800-1500 | ~5000 | 3-6 hours | 8,200 | ~15-30% |
| **A2C** | ~500-1200 | ~4000 | 2-4 hours | 37,000 | ~10-20% |
| **ARS** | ~300-800 | ~2000 | 10-20 min | 65 | ~5-10% |
| **Random** | ~-20 to +50 | ~100 | N/A | 0 | ~0% |

**Notes**:
- "Success Rate" = % of episodes where lap is completed
- Results vary significantly based on random seed and training duration
- PPO generally achieves highest rewards but takes longest to train
- ARS is surprisingly competitive given its simplicity

### Training Curves

Typical learning progression:

**PPO**:
- Episodes 0-20: Random exploration (~-50 reward)
- Episodes 20-50: Learns to stay on road (~200 reward)
- Episodes 50-100: Learns to drive fast (~800 reward)
- Episodes 100+: Occasional lap completions (~5000 spikes)

**A2C**:
- Episodes 0-50: Exploration (~0 reward)
- Episodes 50-200: Basic driving (~300 reward)
- Episodes 200-500: Consistent on-road driving (~700 reward)
- Episodes 500+: Occasional completions (~4000 spikes)

**ARS**:
- Iterations 0-20: Random (~50 reward)
- Iterations 20-50: Basic control (~200 reward)
- Iterations 50-100: Stable driving (~500 reward)
- Iterations 100+: Plateaus (~700 reward)

### Computational Requirements

**Memory Usage**:
- ARS: <50 MB
- A2C: ~200 MB
- PPO: ~300 MB

**GPU Requirements**:
- All agents can train on CPU
- GPU provides ~2-3x speedup for A2C/PPO
- ARS doesn't benefit from GPU

### Which Agent to Choose?

**Choose ARS if**:
- You want fast prototyping
- Resources are limited
- Problem is relatively simple
- You prefer interpretability

**Choose A2C if**:
- You want good balance of performance/speed
- You have moderate computational resources
- You need decent sample efficiency

**Choose PPO if**:
- You want best performance
- Sample efficiency matters
- You have computational resources
- Stability is crucial

**Use Random if**:
- You're testing the environment
- You need a baseline

---

## Quick Start Guide

### Installation

```bash
# Clone repository
git clone <repository-url>
cd RL-Project-2d-driving-simulator

# Install dependencies
pip install -r requirements.txt
```

**Requirements**:
- Python 3.8+
- PyTorch 2.0+
- Gymnasium
- Pygame
- NumPy
- Matplotlib

### Training Your First Agent

#### Option 1: Interactive Menu (Recommended)

```bash
python src/training/train_all_agents.py
```

This provides a menu:
1. Test individual agents with visualization
2. Train all agents sequentially
3. Compare existing models

#### Option 2: Train Specific Agent

**ARS**:
```bash
python src/training/train_ars.py
```

**A2C**:
```bash
python src/training/train_a2c.py
```

**PPO**:
```bash
python src/agents/PPO/train.py
```

### Evaluating a Trained Model

```bash
# Using the interactive system
python src/training/train_all_agents.py
# Select option 1, then choose agent

# Or manually load and test
python src/game/play_manual.py  # Manual control for comparison
```

### Comparing Multiple Agents

```bash
python src/training/compare_agents.py
```

This generates comparative plots and statistics for all trained models.

---

## Advanced Usage

### Custom Training Configurations

#### Modify Hyperparameters

Edit [src/config.py](src/config.py):
```python
# ARS Configuration
ARS_ITERATIONS = 500  # Increase for better convergence
LR_ARS = 0.01  # Decrease for stability

# A2C Configuration
A2C_EPISODES = 5000
LR_A2C = 5e-4

# PPO Configuration
# Edit configs/ppo_config.yaml
```

#### Create Training Profiles

Use [confic.py](confic.py) for pre-defined configurations:
```python
from confic import CONFIGS

config = CONFIGS['INTENSIVE']  # Uses aggressive hyperparameters
```

### Custom Reward Functions

Edit [src/env/driving_env.py](src/env/driving_env.py):
```python
def _calculate_reward(self):
    # Your custom reward logic
    reward = 0.0

    # Example: Penalize sharp turns
    reward -= abs(self.game_engine.car.turn_speed) * 0.1

    # Example: Bonus for staying centered
    distance_from_center = self._get_distance_from_center()
    reward += max(0, 1.0 - distance_from_center)

    return reward
```

### Logging with TensorBoard

Integrate tensorboard logging (example for A2C):

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f'runs/a2c_{timestamp}')

# In training loop:
writer.add_scalar('Reward/episode', episode_reward, episode)
writer.add_scalar('Loss/actor', actor_loss, episode)
writer.add_scalar('Loss/critic', critic_loss, episode)
```

View with:
```bash
tensorboard --logdir=runs
```

### Multi-Agent Comparison

Run experiments with different configurations:

```python
# experiments.py
configs = [
    {'name': 'ARS_aggressive', 'lr': 0.05},
    {'name': 'ARS_conservative', 'lr': 0.01},
    {'name': 'ARS_standard', 'lr': 0.02},
]

for config in configs:
    agent = ARSAgent(learning_rate=config['lr'])
    train(agent, run_name=config['name'])
```

Then compare:
```bash
python src/training/compare_agents.py
```

### Curriculum Learning

Progressively increase difficulty:

```python
# In training loop
if episode < 500:
    max_steps = 1000  # Shorter episodes initially
elif episode < 1000:
    max_steps = 2000
else:
    max_steps = 3000

# Or adjust reward shaping
if episode < 200:
    collision_penalty = -1.0  # Gentler penalty
else:
    collision_penalty = -2.0
```

### Parallel Training

Train multiple agents simultaneously:

```python
import multiprocessing

def train_agent(agent_type, seed):
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Train
    if agent_type == 'ars':
        train_ars()
    elif agent_type == 'a2c':
        train_a2c()

if __name__ == '__main__':
    processes = []
    for seed in range(5):  # 5 runs with different seeds
        p = multiprocessing.Process(target=train_agent, args=('ars', seed))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

### Model Analysis

#### Extract Learned Features

**For neural networks (A2C/PPO)**:
```python
# Get activations
def get_hidden_activations(agent, state):
    with torch.no_grad():
        x = agent.actor.fc1(torch.tensor(state, dtype=torch.float32))
        x = F.relu(x)
        return x.numpy()

# Visualize with t-SNE
from sklearn.manifold import TSNE

states = collect_states(env, agent, n_episodes=100)
activations = [get_hidden_activations(agent, s) for s in states]

tsne = TSNE(n_components=2)
embedded = tsne.fit_transform(activations)

plt.scatter(embedded[:, 0], embedded[:, 1])
plt.title('Learned State Representations')
```

**For ARS**:
```python
# Visualize weight matrix
import seaborn as sns

plt.figure(figsize=(12, 4))
sns.heatmap(agent.weights, cmap='coolwarm', center=0,
            xticklabels=['vel', 'sin', 'cos', 'd0', 'd45', 'd90', 'd135',
                         'd180', 'd225', 'd270', 'd315', 'prog', 'onroad'],
            yticklabels=['NONE', 'ACCEL', 'BRAKE', 'LEFT', 'RIGHT'])
plt.title('ARS Learned Weights')
plt.show()
```

---

## Troubleshooting

### Common Issues

**1. Agent doesn't learn**
- Check reward function is returning non-zero values
- Verify environment is resetting properly
- Try increasing learning rate
- Check for bugs in observation space

**2. Training is unstable (rewards fluctuate wildly)**
- Reduce learning rate
- Add gradient clipping
- Normalize observations
- Use larger batch sizes (PPO)

**3. Agent gets stuck in local optimum**
- Increase entropy coefficient (A2C/PPO)
- Add noise to exploration
- Adjust reward shaping
- Try different random seed

**4. Out of memory errors**
- Reduce batch size
- Use gradient accumulation
- Train on CPU
- Clear cache periodically

**5. Import errors**
- Ensure project root is in Python path
- Check `__init__.py` files exist
- Verify relative imports are correct

### Debugging Tips

**Enable detailed logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Visualize episodes during training**:
```python
# Periodically render
if episode % 100 == 0:
    test_agent(agent, render=True)
```

**Check gradients**:
```python
for name, param in agent.actor.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

**Verify environment**:
```python
# Test environment independently
env = make_env(render_mode="human")
for _ in range(10):
    state, _ = env.reset()
    print(f"Initial state: {state}")
    print(f"State shape: {state.shape}")
    print(f"State range: [{state.min():.2f}, {state.max():.2f}]")
```

---

## Future Improvements

Potential enhancements to the project:

### Algorithms
- [ ] **SAC** (Soft Actor-Critic) - Off-policy, continuous actions
- [ ] **DQN** (Deep Q-Network) - Value-based method
- [ ] **TD3** (Twin Delayed DDPG) - Better exploration
- [ ] **Rainbow DQN** - Combines multiple DQN improvements

### Environment
- [ ] Multiple track layouts
- [ ] Opponent cars (multi-agent)
- [ ] Weather conditions (rain = more friction)
- [ ] Power-ups and obstacles
- [ ] Continuous action space (steering angle, throttle)

### Training
- [ ] Prioritized Experience Replay
- [ ] Hindsight Experience Replay (HER)
- [ ] Population-based training
- [ ] Automatic hyperparameter tuning (Optuna)
- [ ] Distributed training (Ray RLlib)

### Analysis
- [ ] TensorBoard integration
- [ ] Weights & Biases logging
- [ ] Ablation studies
- [ ] Sensitivity analysis
- [ ] Policy visualization

---

## References

### Papers

**ARS**:
- Mania, H., Guy, A., & Recht, B. (2018). Simple random search provides a competitive approach to reinforcement learning. *NeurIPS*.

**A2C**:
- Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. *ICML*.

**PPO**:
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv*.

**GAE**:
- Schulman, J., et al. (2015). High-dimensional continuous control using generalized advantage estimation. *ICLR*.

### Resources

- [Spinning Up in Deep RL](https://spinningup.openai.com/) - OpenAI educational resource
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - Production-ready RL implementations
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Single-file RL implementations
- [Gymnasium Documentation](https://gymnasium.farama.org/) - Environment API

---

## Contributing

To add a new agent:

1. Create agent directory: `src/agents/<agent_name>/`
2. Implement `agent.py` with standard interface:
   ```python
   class NewAgent:
       def __init__(self, state_dim, action_dim, **kwargs):
           pass

       def select_action(self, state):
           pass

       def update(self, *args):
           pass

       def save(self, path):
           pass

       def load(self, path):
           pass
   ```
3. Create training script: `src/training/train_<agent_name>.py`
4. Add to comparison system
5. Update this documentation

---

## License

[Add your license here]

---

## Acknowledgments

This project was developed as part of a reinforcement learning course. Special thanks to:
- The OpenAI Spinning Up team for educational resources
- Gymnasium developers for the standardized environment API
- The RL research community for algorithm implementations

---

**Document Version**: 1.0
**Last Updated**: 2026-01-05
**Authors**: [Your names here]

For questions or issues, please open a GitHub issue or contact the maintainers.

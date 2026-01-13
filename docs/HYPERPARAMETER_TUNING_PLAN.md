# Hyperparameter Tuning Plan for ARS and A2C Agents

This document outlines a systematic approach to finding the best hyperparameter configurations for the ARS and A2C agents on the 2D racing environment.

## Overview

The goal is to find configurations that allow the agents to:
1. Complete at least one full lap
2. Achieve consistent performance across runs
3. Train efficiently (fast convergence)

---

## Current Baseline Configuration

### ARS (from config.py)
| Parameter | Value | Description |
|-----------|-------|-------------|
| learning_rate | 0.02 | Step size for weight updates |
| noise | 0.03 | Perturbation scale |
| num_deltas | 16 | Number of random directions per iteration |
| total_iterations | 500 | Training iterations |

### A2C (from config.py)
| Parameter | Value | Description |
|-----------|-------|-------------|
| learning_rate | 1e-3 | Adam optimizer learning rate |
| gamma | 0.99 | Discount factor |
| entropy_coef | 0.01 | Exploration bonus weight |
| total_episodes | 500 | Training episodes |
| network_size | 128-128 | Hidden layer sizes |

---

## Tuning Strategy

### Phase 1: ARS Hyperparameter Search

ARS has only 3 main hyperparameters, making it easy to tune systematically.

#### Experiment 1.1: Learning Rate
| Config | learning_rate | noise | num_deltas |
|--------|---------------|-------|------------|
| ARS-LR1 | 0.01 | 0.03 | 16 |
| ARS-LR2 | **0.02** (baseline) | 0.03 | 16 |
| ARS-LR3 | 0.05 | 0.03 | 16 |
| ARS-LR4 | 0.1 | 0.03 | 16 |

**Expected outcomes:**
- Lower LR (0.01): Slower but more stable learning
- Higher LR (0.1): Faster but potentially unstable

#### Experiment 1.2: Noise Scale
| Config | learning_rate | noise | num_deltas |
|--------|---------------|-------|------------|
| ARS-N1 | 0.02 | 0.01 | 16 |
| ARS-N2 | 0.02 | **0.03** (baseline) | 16 |
| ARS-N3 | 0.02 | 0.05 | 16 |
| ARS-N4 | 0.02 | 0.1 | 16 |

**Expected outcomes:**
- Lower noise (0.01): Less exploration, fine-tuning mode
- Higher noise (0.1): More exploration, might miss optimal

#### Experiment 1.3: Number of Deltas
| Config | learning_rate | noise | num_deltas |
|--------|---------------|-------|------------|
| ARS-D1 | 0.02 | 0.03 | 8 |
| ARS-D2 | 0.02 | 0.03 | **16** (baseline) |
| ARS-D3 | 0.02 | 0.03 | 32 |
| ARS-D4 | 0.02 | 0.03 | 64 |

**Expected outcomes:**
- Fewer deltas (8): Faster iterations, noisier gradient estimates
- More deltas (64): Better gradient estimates, but slower per iteration

#### Experiment 1.4: Combined Best Parameters
After running 1.1-1.3, combine the best settings from each experiment.

---

### Phase 2: A2C Hyperparameter Search

A2C has more hyperparameters and neural network architecture to tune.

#### Experiment 2.1: Learning Rate
| Config | learning_rate | gamma | entropy_coef | network |
|--------|---------------|-------|--------------|---------|
| A2C-LR1 | 5e-4 | 0.99 | 0.01 | 128-128 |
| A2C-LR2 | **1e-3** (baseline) | 0.99 | 0.01 | 128-128 |
| A2C-LR3 | 3e-3 | 0.99 | 0.01 | 128-128 |
| A2C-LR4 | 1e-2 | 0.99 | 0.01 | 128-128 |

**Expected outcomes:**
- Lower LR (5e-4): Slower but more stable
- Higher LR (1e-2): Might diverge or oscillate

#### Experiment 2.2: Entropy Coefficient
| Config | learning_rate | gamma | entropy_coef | network |
|--------|---------------|-------|--------------|---------|
| A2C-E1 | 1e-3 | 0.99 | 0.001 | 128-128 |
| A2C-E2 | 1e-3 | 0.99 | **0.01** (baseline) | 128-128 |
| A2C-E3 | 1e-3 | 0.99 | 0.05 | 128-128 |
| A2C-E4 | 1e-3 | 0.99 | 0.1 | 128-128 |

**Expected outcomes:**
- Lower entropy (0.001): Less exploration, might get stuck
- Higher entropy (0.1): Too much exploration, slow convergence

#### Experiment 2.3: Discount Factor (Gamma)
| Config | learning_rate | gamma | entropy_coef | network |
|--------|---------------|-------|--------------|---------|
| A2C-G1 | 1e-3 | 0.95 | 0.01 | 128-128 |
| A2C-G2 | 1e-3 | 0.97 | 0.01 | 128-128 |
| A2C-G3 | 1e-3 | **0.99** (baseline) | 0.01 | 128-128 |
| A2C-G4 | 1e-3 | 0.995 | 0.01 | 128-128 |

**Expected outcomes:**
- Lower gamma (0.95): Short-term focus, faster initial learning
- Higher gamma (0.995): Long-term focus, better for lap completion

#### Experiment 2.4: Network Architecture
| Config | learning_rate | network_size | activation |
|--------|---------------|--------------|------------|
| A2C-NET1 | 1e-3 | 64-64 | ReLU |
| A2C-NET2 | 1e-3 | **128-128** (baseline) | ReLU |
| A2C-NET3 | 1e-3 | 256-256 | ReLU |
| A2C-NET4 | 1e-3 | 128-128-128 | ReLU |

**Expected outcomes:**
- Smaller networks: Faster training, might underfit
- Larger networks: More capacity, slower training, might overfit

#### Experiment 2.5: Combined Best Parameters
After running 2.1-2.4, combine the best settings from each experiment.

---

## Recommended Training Protocol

### For Each Configuration:

1. **Run 3 seeds** per configuration for statistical significance
   ```bash
   python ARS/train_ars.py  # Run 3 times
   python A2C/train_a2c.py  # Run 3 times
   ```

2. **Metrics to track:**
   - Best reward achieved
   - Average reward (last 50 episodes/iterations)
   - Number of checkpoints passed
   - Whether a lap was completed
   - Time to first lap completion

3. **Early stopping criteria:**
   - If reward doesn't improve for 100 iterations/episodes
   - If agent consistently completes laps

---

## Implementation: How to Run Experiments

### Method 1: Manual Configuration Changes

Edit `config.py` before each run:

```python
# === ARS Experiment 1.1: Learning Rate Test ===
ARS_LEARNING_RATE = 0.05  # Change this value
ARS_NOISE = 0.03
ARS_NUM_DELTAS = 16
ARS_TOTAL_ITERATIONS = 500
```

### Method 2: Command-Line Override (Recommended)

Create a script `run_experiments.py`:

```python
import subprocess
import os

# ARS Learning Rate experiments
ars_lr_experiments = [
    {"name": "ARS-LR1", "lr": 0.01, "noise": 0.03, "deltas": 16},
    {"name": "ARS-LR2", "lr": 0.02, "noise": 0.03, "deltas": 16},
    {"name": "ARS-LR3", "lr": 0.05, "noise": 0.03, "deltas": 16},
    {"name": "ARS-LR4", "lr": 0.10, "noise": 0.03, "deltas": 16},
]

for exp in ars_lr_experiments:
    print(f"Running experiment: {exp['name']}")
    # Update config.py programmatically or pass as arguments
    subprocess.run(["python", "ARS/train_ars.py"])
```

---

## Quick Start: Recommended First Experiments

If you want to start testing immediately, here are the most impactful experiments:

### ARS Priority Experiments:
1. **ARS-LR3** (lr=0.05): Try higher learning rate first
2. **ARS-D3** (deltas=32): More directions = better gradient

### A2C Priority Experiments:
1. **A2C-G4** (gamma=0.995): Higher discount for long-term rewards
2. **A2C-E3** (entropy=0.05): More exploration initially
3. **A2C-NET3** (256-256): Larger network capacity

---

## Results Tracking Template

Create a spreadsheet or CSV to track results:

| Experiment | Best Reward | Avg Reward (last 50) | Checkpoints | Lap Complete? | Notes |
|------------|-------------|----------------------|-------------|---------------|-------|
| ARS-baseline | | | | | |
| ARS-LR1 | | | | | |
| ARS-LR2 | | | | | |
| ... | | | | | |
| A2C-baseline | | | | | |
| A2C-LR1 | | | | | |
| ... | | | | | |

---

## Advanced Tuning (After Initial Experiments)

### If agents still can't complete a lap:

1. **Increase training duration:**
   - ARS: 1000-2000 iterations
   - A2C: 1000-2000 episodes

2. **Adjust reward shaping:**
   ```python
   # In config.py
   REWARD_CHECKPOINT = 100.0    # Increase checkpoint reward
   REWARD_SPEED = 0.1           # Reduce speed reward (less reward hacking)
   REWARD_OFFROAD = -10.0       # Stronger penalty for going off-road
   ```

3. **Curriculum learning:**
   - Start with shorter episodes
   - Gradually increase difficulty

### If training is unstable:

1. **For A2C:**
   - Lower learning rate
   - Increase gradient clipping (change 0.5 to 1.0)
   - Reduce entropy coefficient

2. **For ARS:**
   - Lower noise
   - Increase num_deltas

---

## Summary

| Agent | Key Hyperparameters | Typical Good Ranges |
|-------|---------------------|---------------------|
| ARS | learning_rate | 0.01 - 0.1 |
| ARS | noise | 0.01 - 0.1 |
| ARS | num_deltas | 16 - 64 |
| A2C | learning_rate | 5e-4 - 3e-3 |
| A2C | entropy_coef | 0.001 - 0.1 |
| A2C | gamma | 0.95 - 0.995 |
| A2C | network_size | 128-128 to 256-256 |

Good luck with the hyperparameter search!

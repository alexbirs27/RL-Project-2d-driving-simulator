"""
ARS (Augmented Random Search) Training Script

This script trains an ARS agent on the driving environment.
ARS is a gradient-free algorithm that explores by adding random noise
to policy weights and comparing the performance of positive vs negative
perturbations.

Training loop:
1. Generate random perturbations (deltas)
2. For each delta, run two episodes: one with +delta, one with -delta
3. Compare rewards and update weights toward better directions
4. Repeat for many iterations

Usage:
    python train_ars.py                     # Train without rendering
    python train_ars.py --render            # Train with rendering every iteration
    python train_ars.py --render --render-freq 5  # Render every 5 iterations
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pygame
from datetime import datetime

# Setup import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import configuration and modules
from config import *
from env import make_env
from ARS.agent import ARSAgent

# === GENERATE UNIQUE TIMESTAMP FOR THIS RUN ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"ars_run_{timestamp}"

print(f"--- NEW RUN: {run_name} ---")

# === SETUP DIRECTORIES ===
logs_dir = os.path.join(project_root, 'logs')
models_dir = os.path.join(project_root, 'models')
plots_dir = os.path.join(project_root, 'plots')

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Unique log file for this run
log_file = os.path.join(logs_dir, f"{run_name}_log.csv")
with open(log_file, "w") as f:
    f.write("iteration,avg_reward,max_reward,laps\n")

# Path for saving the best model
best_model_path = os.path.join(models_dir, f'{run_name}_BEST_weights.npy')


def train_ars_script(render=False, render_freq=1):
    """
    Main training function for ARS agent.

    Args:
        render: If True, enable visualization during training
        render_freq: Render every N iterations (e.g., 5 = render 1 out of 5)
    """
    # Initialize the environment (no rendering by default for faster training)
    env = make_env(render_mode=None)

    # Get dimensions from environment (not hardcoded)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize the ARS agent with hyperparameters from config
    print(f"Initializing ARS (state_dim={state_dim}, action_dim={action_dim}, StepSize={ARS_LEARNING_RATE})...")
    agent = ARSAgent(state_dim=state_dim, action_dim=action_dim, learning_rate=ARS_LEARNING_RATE)

    # Number of training iterations from config
    num_iterations = ARS_TOTAL_ITERATIONS

    # History for plotting
    history_avg_reward = []
    history_max_reward = []
    history_laps = []

    # Track best performance for checkpoint saving
    best_reward = -float('inf')
    total_laps = 0  # Total laps across all training

    print(f"--- Starting ARS Training ({num_iterations} iterations) ---")

    for i in range(num_iterations):
        # Enable rendering for this iteration if requested
        should_render = render and (i % render_freq == 0)
        if should_render:
            env.close()
            env = make_env(render_mode="human")

        # === Step 1: Generate Random Perturbations ===
        # Each delta is a random matrix with the same shape as the policy weights
        deltas = [np.random.randn(*agent.weights.shape) for _ in range(agent.num_deltas)]
        rollouts = []
        rewards_list = []
        iteration_laps = 0  # Laps completed in this iteration

        # === Step 2: Evaluate Each Perturbation (+ and -) ===
        for delta in deltas:
            # Run episode with weights + noise*delta
            r_pos, lap_pos = run_episode(env, agent, delta, direction="plus", should_render=should_render)
            # Run episode with weights - noise*delta
            r_neg, lap_neg = run_episode(env, agent, delta, direction="minus", should_render=should_render)

            rollouts.append((r_pos, r_neg, delta))
            rewards_list.extend([r_pos, r_neg])

            # Count laps
            if lap_pos:
                iteration_laps += 1
            if lap_neg:
                iteration_laps += 1

        # Disable rendering after iteration if it was enabled
        if should_render and i < num_iterations - 1:
            env.close()
            env = make_env(render_mode=None)

        # === Step 3: Update Agent Weights ===
        # Compute standard deviation of rewards for normalization
        sigma_r = np.std(rewards_list) if np.std(rewards_list) > 0 else 1
        agent.update(rollouts, sigma_r)

        # Compute statistics for this iteration
        avg_reward = np.mean(rewards_list)
        max_reward = np.max(rewards_list)

        history_avg_reward.append(avg_reward)
        history_max_reward.append(max_reward)
        history_laps.append(iteration_laps)
        total_laps += iteration_laps

        # === Save Checkpoint if New Best ===
        # For ARS, we track the maximum reward found in any direction
        if max_reward > best_reward:
            best_reward = max_reward
            np.save(best_model_path, agent.weights)
            print(f"!!! NEW RECORD: {best_reward:.2f} -> Model saved.")

        # Progress logging
        print(f"Iteration {i+1}/{num_iterations} | Avg: {avg_reward:.2f} | Max: {max_reward:.2f} | Best: {best_reward:.2f} | Laps: {iteration_laps} (Total: {total_laps})")

        # Save to CSV log
        with open(log_file, "a") as f:
            f.write(f"{i+1},{avg_reward},{max_reward},{iteration_laps}\n")

    # === Save Final Model ===
    final_path = os.path.join(models_dir, f'{run_name}_FINAL_weights.npy')
    np.save(final_path, agent.weights)
    print(f"Training complete. Best Reward: {best_reward:.2f} | Total Laps: {total_laps}")
    print(f"Final model saved to: {final_path}")

    # === Generate Training Plot ===
    print("Generating plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Rewards
    ax1.plot(history_avg_reward, label='Average Reward', color='blue', alpha=0.7)
    ax1.plot(history_max_reward, label='Max Reward', color='green', linestyle='--')
    ax1.axhline(y=best_reward, color='red', linestyle=':', label=f'Best Ever ({best_reward:.0f})')
    ax1.set_title(f'ARS Training Rewards: {run_name}')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Laps per iteration
    ax2.bar(range(len(history_laps)), history_laps, color='purple', alpha=0.7)
    ax2.set_title(f'Laps Completed per Iteration (Total: {total_laps})')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Laps')
    ax2.grid(True, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'{run_name}_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to: {plot_path}")

    env.close()


def run_episode(env, agent, delta, direction, should_render=False):
    """
    Run a single episode with the given perturbation.

    Args:
        env: The gym environment
        agent: The ARS agent
        delta: Random perturbation matrix
        direction: "plus" or "minus" - which direction to perturb weights
        should_render: If True, render the episode

    Returns:
        total_reward: Sum of rewards collected during the episode
        lap_completed: Whether the agent completed a lap
    """
    state, _ = env.reset()
    total_reward = 0
    done = False
    lap_completed = False

    # Safety limit to prevent infinite loops
    steps = 0
    max_steps = 2000

    while not done and steps < max_steps:
        # Handle pygame events to prevent window freeze
        if should_render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\nTraining interrupted by user")
                    env.close()
                    return total_reward, lap_completed

        # Select action using perturbed weights
        action = agent.select_action(state, delta, direction)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        # Check if lap was completed (terminated means lap complete)
        if terminated and info.get('lap_complete', False):
            lap_completed = True

        if should_render:
            env.render()

    return total_reward, lap_completed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ARS agent")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--render-freq", type=int, default=1, help="Render every N iterations (default: 1)")
    args = parser.parse_args()

    train_ars_script(render=args.render, render_freq=args.render_freq)

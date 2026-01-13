"""
Training loop:
1. Generate random perturbations (deltas)
2. For each delta, run two episodes: one with +delta, one with -delta
3. Compare rewards and update weights toward better directions
4. Repeat for many iterations
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pygame
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import configuration and modules
from config import *
from env import make_env
from ARS.agent import ARSAgent

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"ars_run_{timestamp}"

print(f"--- NEW RUN: {run_name} ---")

logs_dir = os.path.join(current_dir, 'logs')
models_dir = os.path.join(current_dir, 'models')
plots_dir = os.path.join(current_dir, 'plots')

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

log_file = os.path.join(logs_dir, f"{run_name}_log.csv")
with open(log_file, "w") as f:
    f.write("iteration,avg_reward,max_reward,laps\n")

# Path for saving the best model (using .npz for full checkpoint)
best_model_path = os.path.join(models_dir, f'{run_name}_BEST_weights.npz')


def train_ars_script(render=False, render_freq=1):
   
    # Initialize the environment (no rendering by default for faster training)
    env = make_env(render_mode=None)

    # Get dimensions from environment (not hardcoded)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    num_iterations = ARS_TOTAL_ITERATIONS

    # Initialize the ARS agent with hyperparameters from config
    # top_k = num_deltas // 2 is a common choice (use best half of directions)
    print(f"Initializing ARS (state_dim={state_dim}, action_dim={action_dim}, StepSize={ARS_LEARNING_RATE})...")
    agent = ARSAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=ARS_LEARNING_RATE,
        num_deltas=16,
        top_k=8,  # Use best 8 out of 16 deltas
        total_iterations=num_iterations,
        normalize_states=True
    )

    history_avg_reward = []
    history_max_reward = []
    history_laps = []

    # Track best performance for checkpoint saving
    best_reward = -float('inf')
    total_laps = 0 

    print(f"--- Starting ARS Training ({num_iterations} iterations) ---")

    for i in range(num_iterations):
        # Enable rendering for this iteration if requested
        should_render = render and (i % render_freq == 0)
        if should_render:
            env.close()
            env = make_env(render_mode="human")

        # Each delta is a random matrix with the same shape as the policy weights
        deltas = [np.random.randn(*agent.weights.shape) for _ in range(agent.num_deltas)]
        rollouts = []
        rewards_list = []
        iteration_laps = 0  
        
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

        
        # Tracking the maximum reward found in any direction
        if max_reward > best_reward:
            best_reward = max_reward
            agent.save(best_model_path)
            print(f"!!! NEW RECORD: {best_reward:.2f} -> Model saved.")

        # Progress logging
        print(f"Iteration {i+1}/{num_iterations} | Avg: {avg_reward:.2f} | Max: {max_reward:.2f} | Best: {best_reward:.2f} | Laps: {iteration_laps} (Total: {total_laps})")

        # Save to CSV log
        with open(log_file, "a") as f:
            f.write(f"{i+1},{avg_reward},{max_reward},{iteration_laps}\n")

    # Saving final model
    final_path = os.path.join(models_dir, f'{run_name}_FINAL_weights.npz')
    agent.save(final_path)
    print(f"Training complete. Best Reward: {best_reward:.2f} | Total Laps: {total_laps}")
    print(f"Final model saved to: {final_path}")

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
        # update_stats=True to build normalization statistics during training
        action = agent.select_action(state, delta, direction, update_stats=True)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        # Check if lap was completed 
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

"""
A2C (Advantage Actor-Critic) Training Script

This script trains an A2C agent on the driving environment.
A2C is a policy gradient algorithm that uses two neural networks:
- Actor: Learns what action to take in each state
- Critic: Learns to estimate how good each state is

Training loop:
1. Run an episode, collecting states, actions, rewards
2. Compute discounted returns (how much total reward each step led to)
3. Compute advantages (how much better the action was vs expected)
4. Update both networks using gradient descent

Usage:
    python train_a2c.py                     # Train without rendering
    python train_a2c.py --render            # Train with rendering every episode
    python train_a2c.py --render --render-freq 10  # Render every 10 episodes
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
from A2C.agent import A2CAgent

# === GENERATE UNIQUE TIMESTAMP FOR THIS RUN ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"a2c_run_{timestamp}"

print(f"--- NEW RUN: {run_name} ---")

# === SETUP DIRECTORIES ===
logs_dir = os.path.join(project_root, 'logs')
models_dir = os.path.join(project_root, 'models')
plots_dir = os.path.join(project_root, 'plots')

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Initialize CSV log file
log_file = os.path.join(logs_dir, f"{run_name}_log.csv")
with open(log_file, "w") as f:
    f.write("episode,reward,loss,lap\n")

# Path for saving the best model
best_model_path = os.path.join(models_dir, f'{run_name}_BEST_model.pt')


def train_a2c_script(render=False, render_freq=1):
    """
    Main training function for A2C agent.

    Args:
        render: If True, enable visualization during training
        render_freq: Render every N episodes (e.g., 5 = render 1 out of 5)
    """
    # Initialize the environment (no rendering by default for faster training)
    env = make_env(render_mode=None)

    # Get dimensions from environment (not hardcoded)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize A2C agent with hyperparameters from config
    print(f"Initializing A2C (state_dim={state_dim}, action_dim={action_dim}, LR={A2C_LEARNING_RATE})...")
    agent = A2CAgent(state_dim=state_dim, action_dim=action_dim, lr=A2C_LEARNING_RATE)

    # Number of training episodes from config
    num_episodes = A2C_TOTAL_EPISODES

    # History for plotting
    history_rewards = []
    history_loss = []
    history_laps = []  # Track which episodes had lap completions

    # Track best performance for checkpoint saving
    best_reward = -float('inf')
    total_laps = 0  # Total laps across all training

    print(f"--- Starting A2C Training ({num_episodes} episodes) ---")

    for episode in range(num_episodes):
        # Enable rendering for this episode if requested
        should_render = render and (episode % render_freq == 0)
        if should_render:
            env.close()
            env = make_env(render_mode="human")

        # Reset environment for new episode
        state, _ = env.reset()
        done = False

        # Episode data collection
        rewards = []       # Rewards at each step
        log_probs = []     # Log probabilities of actions taken
        states = []        # States visited
        dones = []         # Done flags

        total_reward = 0
        lap_completed = False  # Track if lap was completed this episode

        # === Run Episode ===
        while not done:
            # Handle pygame events to prevent window freeze
            if should_render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nTraining interrupted by user")
                        env.close()
                        return

            # Get action from agent (includes exploration via sampling)
            action, log_prob = agent.act(state)

            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Check if lap was completed
            if terminated and info.get('lap_complete', False):
                lap_completed = True

            # Store experience for learning
            states.append(state)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)

            total_reward += reward
            state = next_state

            if should_render:
                env.render()

        # Disable rendering after episode if it was enabled
        if should_render and episode < num_episodes - 1:
            env.close()
            env = make_env(render_mode=None)

        # === Update Agent ===
        # Pass all collected data to the agent for learning
        loss = agent.update(rewards, log_probs, states, dones, next_state)

        # Store history for plotting
        history_rewards.append(total_reward)
        history_loss.append(loss)
        history_laps.append(1 if lap_completed else 0)

        # Update total laps count
        if lap_completed:
            total_laps += 1

        # === Save Checkpoint if New Best ===
        # Save model only when it breaks the record
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(best_model_path)
            print(f"!!! NEW RECORD: {best_reward:.2f} -> Model saved to {os.path.basename(best_model_path)}")

        # Progress logging every 10 episodes
        if (episode + 1) % 10 == 0:
            lap_str = "[LAP!]" if lap_completed else ""
            print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward:.2f} | Loss: {loss:.4f} | Best: {best_reward:.2f} | Laps: {total_laps} {lap_str}")

        # Save to CSV log
        with open(log_file, "a") as f:
            f.write(f"{episode+1},{total_reward},{loss},{1 if lap_completed else 0}\n")

    # === Save Final Model ===
    # Save the final model even if it's not the best (useful for comparison)
    final_path = os.path.join(models_dir, f'{run_name}_FINAL_model.pt')
    agent.save(final_path)
    print(f"Training complete. Best Reward: {best_reward:.2f} | Total Laps: {total_laps}")

    # === Generate Training Plots ===
    print("Generating plots...")

    # Compute moving average for smoother reward visualization
    window_size = 20
    if len(history_rewards) >= window_size:
        moving_avg = np.convolve(history_rewards, np.ones(window_size)/window_size, mode='valid')
    else:
        moving_avg = history_rewards

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot 1: Rewards
    ax1.plot(history_rewards, label='Episode Reward', alpha=0.3, color='blue')
    ax1.plot(range(len(moving_avg)), moving_avg, label=f'Moving Avg ({window_size})', color='red', linewidth=2)
    ax1.axhline(y=best_reward, color='green', linestyle='--', label=f'Best ({best_reward:.0f})')

    ax1.set_title(f'A2C Rewards: {run_name}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Loss
    ax2.plot(history_loss, label='Training Loss', color='orange')
    ax2.set_title(f'A2C Loss: {run_name}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Cumulative Laps
    cumulative_laps = np.cumsum(history_laps)
    ax3.plot(cumulative_laps, label='Cumulative Laps', color='purple', linewidth=2)
    ax3.scatter([i for i, lap in enumerate(history_laps) if lap == 1],
                [cumulative_laps[i] for i, lap in enumerate(history_laps) if lap == 1],
                color='gold', s=50, zorder=5, label='Lap Completed')
    ax3.set_title(f'Laps Completed (Total: {total_laps})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Cumulative Laps')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'{run_name}_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to: {plot_path}")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train A2C agent")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--render-freq", type=int, default=1, help="Render every N episodes (default: 1)")
    args = parser.parse_args()

    train_a2c_script(render=args.render, render_freq=args.render_freq)

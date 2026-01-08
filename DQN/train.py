import numpy as np                      # for easy work with vectors/matrices and random/shuffle
import torch                            # framework for neural networks and tensor operations
import matplotlib.pyplot as plt
import pygame                           # for handling pygame events during rendering

import sys
import os

# Add parent directory to path to import general env and agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import make_env
from DQN.agent import DQNAgent


# Main Training Loop
def train(render: bool = False):
    render_mode = "human" if render else None
    env = make_env(render_mode=render_mode, max_steps=10000)  # Longer track needs more steps

    agent = DQNAgent(
        state_dim=8,   # Hybrid: 5 rays + centerline distance + centerline angle + velocity
        action_dim=9,
        gamma=0.99,
        lr=1e-4,              # Even lower learning rate for stability
        buffer_size=30000,    # Larger buffer for longer episodes
        batch_size=128,        # Larger batch size
        epsilon_start=1.0,     # Start with full exploration for complex track
        epsilon_end=0.05,      # Higher minimum - always explore a bit
        epsilon_decay=20000,   # MUCH slower decay - explore for 200+ episodes
        tau=0.003              # Slower target network updates
    )


    episodes = 1000  # Increased for longer F1Tenth track
    reward_history = []
    lap_completed_history = []  # Track lap completions
    lap_time_history = []       # Track lap times when completed
    offroad_steps = []          # Track off-road rate
    best_reward = -float('inf') # For checkpointing

    for episode in range(episodes):
        # Decay learning rate to prevent catastrophic forgetting
        current_lr = agent.update_learning_rate(episode, episodes)

        state, _ = env.reset()
        ep_reward = 0
        done = False
        ep_offroad = 0
        ep_steps = 0

        while not done:
            ep_steps += 1
            # Handle pygame events to prevent window freeze
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nTraining interrupted by user")
                        env.close()
                        return agent, reward_history

            # Agent selects action using epsilon-greedy
            action = agent.act(state, training=True)

            # Execute action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track off-road steps
            if not info["on_road"]:
                ep_offroad += 1

            # Store transition in replay buffer
            agent.store(state, action, reward, next_state, done)

            # Train the agent (sample from buffer and update Q-network)
            agent.train()

            # Update state and accumulate reward
            state = next_state
            ep_reward += reward

            if render:
                env.render()

        # Episode finished
        reward_history.append(ep_reward)

        # Track lap completion
        lap_completed_history.append(1 if terminated else 0)

        # Track lap time if completed
        if terminated:
            lap_time_history.append(info["lap_time"])

        # Track off-road rate
        offroad_rate = ep_offroad / ep_steps if ep_steps > 0 else 0
        offroad_steps.append(offroad_rate)

        # Calculate metrics
        if len(reward_history) >= 10:
            mean_reward = np.mean(reward_history[-10:])
        else:
            mean_reward = np.mean(reward_history)

        lap_completion_rate = np.mean(lap_completed_history[-100:]) if lap_completed_history else 0.0
        avg_lap_time = np.mean(lap_time_history[-50:]) if lap_time_history else None
        avg_offroad_rate = np.mean(offroad_steps[-100:]) if offroad_steps else 0.0

        epsilon = agent.get_epsilon()

        # Detailed logging
        log_msg = (
            f"[Episode {episode}] "
            f"Reward: {ep_reward:.2f} | "
            f"Mean Reward (last 10): {mean_reward:.2f} | "
            f"Lap completion: {lap_completion_rate:.2%} | "
        )

        if avg_lap_time is not None:
            log_msg += f"Avg lap time: {avg_lap_time:.2f}s | "
        else:
            log_msg += "Avg lap time: N/A | "

        log_msg += f"Off-road: {avg_offroad_rate:.2%} | Epsilon: {epsilon:.3f} | LR: {current_lr:.2e}"
        print(log_msg)

        # Save checkpoint if best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save({
                'episode': episode,
                'q_network_state_dict': agent.q_network.state_dict(),
                'target_network_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'reward': mean_reward,
            }, 'DQN/best_model.pt')
            print(f"  -> New best model saved! (reward: {mean_reward:.2f})")

    # Plot training performance with multiple metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Episode Rewards
    axes[0, 0].plot(reward_history, label="Episode Reward", alpha=0.4)
    if len(reward_history) > 20:
        moving_avg = np.convolve(reward_history, np.ones(20) / 20, mode="valid")
        axes[0, 0].plot(range(len(moving_avg)), moving_avg, label="Moving Average (20)", linewidth=2)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].set_title("DQN Episode Rewards")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Lap Completion Rate
    if lap_completed_history:
        # Calculate rolling average
        window = 50
        completion_rates = []
        for i in range(len(lap_completed_history)):
            start = max(0, i - window)
            completion_rates.append(np.mean(lap_completed_history[start:i+1]))
        axes[0, 1].plot(completion_rates, label="Lap Completion Rate", color='green')
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Completion Rate")
        axes[0, 1].set_title("Lap Completion Rate (Rolling Avg)")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

    # Plot 3: Lap Times (when completed)
    if lap_time_history:
        axes[1, 0].plot(lap_time_history, label="Lap Time", color='orange', alpha=0.6)
        if len(lap_time_history) > 10:
            lap_time_avg = np.convolve(lap_time_history, np.ones(10) / 10, mode="valid")
            axes[1, 0].plot(range(len(lap_time_avg)), lap_time_avg, label="Moving Average (10)", linewidth=2)
        axes[1, 0].set_xlabel("Completed Laps")
        axes[1, 0].set_ylabel("Time (seconds)")
        axes[1, 0].set_title("Lap Times (Completed Laps Only)")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Plot 4: Off-road Rate
    if offroad_steps:
        axes[1, 1].plot(offroad_steps, label="Off-road Rate", color='red', alpha=0.4)
        if len(offroad_steps) > 20:
            offroad_avg = np.convolve(offroad_steps, np.ones(20) / 20, mode="valid")
            axes[1, 1].plot(range(len(offroad_avg)), offroad_avg, label="Moving Average (20)", linewidth=2)
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Off-road Rate")
        axes[1, 1].set_title("Off-road Rate per Episode")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("DQN/dqn_training_metrics.png")
    print(f"\nTraining metrics saved to DQN/dqn_training_metrics.png")
    plt.show()

    return agent, reward_history


if __name__ == "__main__":
    import sys
    render = "--render" in sys.argv
    train(render=render)

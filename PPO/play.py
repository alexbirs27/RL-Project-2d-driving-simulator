"""
Load and play with a trained PPO agent.

Usage:
    python3 -m PPO.play --model PPO/best_model.pt
    python3 -m PPO.play --model PPO/best_model.pt --episodes 5
"""

import torch
import numpy as np
import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import make_env
from PPO.agent import PPOAgent
from config import OBS_DIM, ACTION_DIM


def play_agent(model_path: str, num_episodes: int = 1):
    """
    Load a trained PPO agent and watch it play.

    Args:
        model_path: Path to the saved model (.pt file)
        num_episodes: Number of episodes to run
    """
    # Create environment with rendering
    env = make_env(render_mode="human")

    # Create agent
    agent = PPOAgent(state_dim=OBS_DIM, action_dim=ACTION_DIM)

    # Load the trained model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, weights_only=False)  # Set weights_only=False for compatibility
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])

    if 'epoch' in checkpoint:
        print(f"Model trained for {checkpoint['epoch']} epochs")
    if 'reward' in checkpoint:
        print(f"Best reward: {checkpoint['reward']:.2f}")

    print(f"\nRunning {num_episodes} episode(s)...\n")

    # Run episodes
    episode_rewards = []
    episode_checkpoints = []
    episode_completions = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # Agent selects action (deterministic - use mean of distribution)
            action, _, _ = agent.act(state)

            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state
            steps += 1

            # Render
            env.render()

        # Episode finished
        episode_rewards.append(episode_reward)
        episode_checkpoints.append(info['checkpoints_visited'])
        episode_completions.append(1 if terminated else 0)

        print(f"Episode {episode + 1}/{num_episodes}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Checkpoints: {info['checkpoints_visited']}")
        print(f"  Completed: {'Yes' if terminated else 'No'}")
        if terminated:
            print(f"  Lap time: {info['lap_time']:.2f}s")
        print(f"  Steps: {steps}")
        print()

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Checkpoints: {np.mean(episode_checkpoints):.1f}")
    print(f"Completion Rate: {np.mean(episode_completions) * 100:.1f}%")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play with a trained PPO agent")
    parser.add_argument("--model", type=str, default="PPO/best_model.pt",
                        help="Path to saved model (default: PPO/best_model.pt)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run (default: 1)")
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("\nAvailable models:")
        if os.path.exists("PPO/best_model.pt"):
            print("  - PPO/best_model.pt")
        if os.path.exists("PPO/final_model.pt"):
            print("  - PPO/final_model.pt")
        sys.exit(1)

    play_agent(args.model, args.episodes)

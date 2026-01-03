import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import os

from environment import RacingEnv
from agent import A2CAgent


class TrainingLogger:
    """Helper class to track and visualize training progress."""
    
    def __init__(self, window_size=100):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_progress = []
        self.moving_avg_rewards = []
        self.window_size = window_size
        
    def log_episode(self, reward: float, length: int, progress: float):
        """Log an episode's statistics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_progress.append(progress)
        
        # Calculate moving average
        if len(self.episode_rewards) >= self.window_size:
            avg = np.mean(self.episode_rewards[-self.window_size:])
            self.moving_avg_rewards.append(avg)
        else:
            self.moving_avg_rewards.append(np.mean(self.episode_rewards))
    
    def plot_training_progress(self, save_path='training_progress.png'):
        """Create a comprehensive training progress plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        axes[0, 0].plot(self.moving_avg_rewards, label=f'Moving Avg ({self.window_size})', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Progress on track
        axes[1, 0].plot(self.episode_progress)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Progress (0-1)')
        axes[1, 0].set_title('Track Progress')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Reward distribution (histogram)
        axes[1, 1].hist(self.episode_rewards[-1000:], bins=50, edgecolor='black')
        axes[1, 1].set_xlabel('Total Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Reward Distribution (Last 1000 Episodes)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Training progress plot saved to {save_path}")


def train_a2c(
    num_episodes: int = 5000,
    max_steps_per_episode: int = 3000,
    update_frequency: int = 10,
    save_frequency: int = 100,
    render_frequency: int = 50,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    save_dir: str = 'models'
):
    """
    Train an A2C agent on the racing environment.
    
    Args:
        num_episodes: Total number of training episodes
        max_steps_per_episode: Maximum steps per episode
        update_frequency: Update the network every N episodes
        save_frequency: Save model every N episodes
        render_frequency: Render environment every N episodes
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        save_dir: Directory to save models and plots
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = RacingEnv(render_mode=None, max_steps=max_steps_per_episode)
    agent = A2CAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=learning_rate,
        gamma=gamma
    )
    
    # Initialize logger
    logger = TrainingLogger(window_size=100)
    
    # Training buffers for batch updates
    states_buffer = []
    actions_buffer = []
    rewards_buffer = []
    values_buffer = []
    log_probs_buffer = []
    dones_buffer = []
    
    # Statistics
    best_reward = -float('inf')
    recent_rewards = deque(maxlen=100)
    
    print("=" * 60)
    print(f"Starting A2C Training")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print(f"Device: {agent.device}")
    print("=" * 60)
    
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_done = False
        max_progress = 0
        
        # Render every N episodes
        should_render = (episode % render_frequency == 0)
        if should_render:
            env.engine.renderer.enabled = True
        
        while not episode_done:
            # Select action
            action, log_prob, value = agent.select_action(state)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_done = terminated or truncated
            
            # Store experience
            states_buffer.append(state)
            actions_buffer.append(action)
            rewards_buffer.append(reward)
            values_buffer.append(value)
            log_probs_buffer.append(log_prob)
            dones_buffer.append(episode_done)
            
            # Update tracking
            episode_reward += reward
            episode_steps += 1
            max_progress = max(max_progress, info['progress'])
            state = next_state
            
            # Render if enabled
            if should_render:
                env.render()
        
        # Disable rendering after episode
        if should_render:
            env.engine.renderer.enabled = False
        
        # Log episode
        recent_rewards.append(episode_reward)
        logger.log_episode(episode_reward, episode_steps, max_progress)
        
        # Update agent every N episodes
        if episode % update_frequency == 0:
            stats = agent.train_step(
                states_buffer,
                actions_buffer,
                rewards_buffer,
                values_buffer,
                log_probs_buffer,
                dones_buffer
            )
            
            # Clear buffers
            states_buffer = []
            actions_buffer = []
            rewards_buffer = []
            values_buffer = []
            log_probs_buffer = []
            dones_buffer = []
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(recent_rewards)
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed
            
            print(f"Episode {episode:5d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Avg(100): {avg_reward:8.2f} | "
                  f"Progress: {max_progress:.2f} | "
                  f"Steps: {episode_steps:4d} | "
                  f"EPS: {eps_per_sec:.2f}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(save_dir, 'best_model.pt'))
        
        # Save checkpoint
        if episode % save_frequency == 0:
            agent.save(os.path.join(save_dir, f'checkpoint_ep{episode}.pt'))
            logger.plot_training_progress(
                os.path.join(save_dir, f'progress_ep{episode}.png')
            )
    
    # Final save
    agent.save(os.path.join(save_dir, 'final_model.pt'))
    logger.plot_training_progress(os.path.join(save_dir, 'final_progress.png'))
    
    elapsed_time = time.time() - start_time
    print("=" * 60)
    print(f"Training Complete!")
    print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final avg reward (100 eps): {np.mean(recent_rewards):.2f}")
    print("=" * 60)
    
    env.close()


def evaluate_agent(model_path: str, num_episodes: int = 10, render: bool = True):
    """
    Evaluate a trained agent.
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
    """
    # Initialize environment and agent
    render_mode = 'human' if render else None
    env = RacingEnv(render_mode=render_mode, max_steps=3000)
    agent = A2CAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    # Load trained model
    agent.load(model_path)
    
    print("=" * 60)
    print(f"Evaluating model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)
    
    episode_rewards = []
    episode_progresses = []
    lap_completions = 0
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_done = False
        max_progress = 0
        
        while not episode_done:
            # Select action (deterministic)
            action, _, _ = agent.select_action(state, deterministic=True)
            
            # Execute action
            state, reward, terminated, truncated, info = env.step(action)
            episode_done = terminated or truncated
            
            episode_reward += reward
            max_progress = max(max_progress, info['progress'])
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_progresses.append(max_progress)
        
        if info.get('lap_complete', False):
            lap_completions += 1
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
              f"Progress = {max_progress:.2f}, "
              f"Lap Complete = {info.get('lap_complete', False)}")
    
    print("=" * 60)
    print(f"Evaluation Results:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Progress: {np.mean(episode_progresses):.2f}")
    print(f"Lap Completions: {lap_completions}/{num_episodes}")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    # Train the agent
    train_a2c(
        num_episodes=5000,
        max_steps_per_episode=3000,
        update_frequency=10,
        save_frequency=100,
        render_frequency=100,
        learning_rate=1e-4,
        gamma=0.99,
        save_dir='models'
    )
    
    # Evaluate the best model
    # evaluate_agent('models/best_model.pt', num_episodes=5, render=True) 
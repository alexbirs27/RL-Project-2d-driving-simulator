import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
from typing import Dict, List, Any, Optional


class TrainingLogger:
    """Advanced logger for tracking and visualizing training progress."""

    def __init__(self, agent_name: str, save_dir: str, window_size: int = 100):
        self.agent_name = agent_name
        self.save_dir = save_dir
        self.window_size = window_size

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_progress = []
        self.moving_avg_rewards = []
        self.timestamps = []

        # Additional metrics
        self.best_reward = -float('inf')
        self.total_steps = 0
        self.start_time = time.time()

        # JSON log file
        self.log_file = os.path.join(save_dir, 'training_log.json')
        self.metrics_history = []

    def log_episode(self, episode: int, reward: float, length: int,
                   progress: float, info: Dict[str, Any] = None):
        """Log statistics for a completed episode."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_progress.append(progress)
        self.timestamps.append(time.time() - self.start_time)
        self.total_steps += length

        # Calculate moving average
        if len(self.episode_rewards) >= self.window_size:
            avg = np.mean(self.episode_rewards[-self.window_size:])
        else:
            avg = np.mean(self.episode_rewards)
        self.moving_avg_rewards.append(avg)

        # Update best reward
        if reward > self.best_reward:
            self.best_reward = reward

        # Create metrics dict
        metrics = {
            'episode': episode,
            'reward': float(reward),
            'length': int(length),
            'progress': float(progress),
            'avg_reward': float(avg),
            'best_reward': float(self.best_reward),
            'total_steps': self.total_steps,
            'timestamp': self.timestamps[-1]
        }

        if info:
            metrics.update({k: float(v) if isinstance(v, (int, float, np.number)) else v
                          for k, v in info.items()})

        self.metrics_history.append(metrics)

        # Save to JSON every 10 episodes
        if episode % 10 == 0:
            self._save_json()

    def _save_json(self):
        """Save metrics to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump({
                'agent_name': self.agent_name,
                'metrics': self.metrics_history
            }, f, indent=2)

    def plot_training_progress(self, save_path: Optional[str] = None):
        """Create comprehensive training progress visualization."""
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'training_progress.png')

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{self.agent_name} Training Progress', fontsize=16)

        episodes = list(range(1, len(self.episode_rewards) + 1))

        # 1. Episode rewards with moving average
        axes[0, 0].plot(episodes, self.episode_rewards, alpha=0.3, label='Episode Reward', color='blue')
        axes[0, 0].plot(episodes, self.moving_avg_rewards,
                       label=f'Moving Avg ({self.window_size})', linewidth=2, color='red')
        axes[0, 0].axhline(y=self.best_reward, color='green', linestyle='--',
                          label=f'Best: {self.best_reward:.2f}')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Rewards Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Episode lengths
        axes[0, 1].plot(episodes, self.episode_lengths, color='purple')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Track progress
        axes[0, 2].plot(episodes, self.episode_progress, color='orange')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Progress (0-1)')
        axes[0, 2].set_title('Track Progress')
        axes[0, 2].axhline(y=1.0, color='green', linestyle='--', label='Complete Lap')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Reward distribution
        recent_rewards = self.episode_rewards[-min(1000, len(self.episode_rewards)):]
        axes[1, 0].hist(recent_rewards, bins=50, edgecolor='black', color='skyblue')
        axes[1, 0].axvline(x=np.mean(recent_rewards), color='red', linestyle='--',
                          label=f'Mean: {np.mean(recent_rewards):.2f}')
        axes[1, 0].set_xlabel('Total Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Reward Distribution (Last {len(recent_rewards)} Episodes)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Learning curve (reward vs steps)
        cumulative_steps = np.cumsum(self.episode_lengths)
        axes[1, 1].plot(cumulative_steps, self.moving_avg_rewards, color='green')
        axes[1, 1].set_xlabel('Total Steps')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].set_title('Learning Curve')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Statistics summary
        axes[1, 2].axis('off')
        stats_text = f"""
        Training Statistics
        {'='*40}
        Agent: {self.agent_name}
        Total Episodes: {len(self.episode_rewards)}
        Total Steps: {self.total_steps:,}

        Best Reward: {self.best_reward:.2f}
        Recent Avg ({self.window_size}): {self.moving_avg_rewards[-1]:.2f}
        Overall Avg: {np.mean(self.episode_rewards):.2f}

        Avg Episode Length: {np.mean(self.episode_lengths):.1f}
        Max Progress: {max(self.episode_progress):.2f}

        Training Time: {self.timestamps[-1]/60:.1f} min
        Episodes/min: {len(self.episode_rewards)/(self.timestamps[-1]/60):.2f}
        """
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                       verticalalignment='center')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training progress plot saved to {save_path}")

    def print_episode_summary(self, episode: int, frequency: int = 10):
        """Print episode summary to console."""
        if episode % frequency == 0:
            recent_rewards = self.episode_rewards[-self.window_size:]
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(self.episode_lengths[-self.window_size:])
            avg_progress = np.mean(self.episode_progress[-self.window_size:])
            eps_per_sec = len(self.episode_rewards) / (time.time() - self.start_time)

            print(f"[{self.agent_name}] Episode {episode:5d} | "
                  f"Reward: {self.episode_rewards[-1]:8.2f} | "
                  f"Avg({self.window_size}): {avg_reward:8.2f} | "
                  f"Progress: {self.episode_progress[-1]:.2f} | "
                  f"Steps: {self.episode_lengths[-1]:4d} | "
                  f"EPS: {eps_per_sec:.2f}")

    def save_checkpoint_info(self, episode: int, model_path: str):
        """Save checkpoint information."""
        checkpoint_info = {
            'episode': episode,
            'model_path': model_path,
            'reward': self.episode_rewards[-1],
            'avg_reward': self.moving_avg_rewards[-1],
            'best_reward': self.best_reward,
            'timestamp': datetime.now().isoformat()
        }

        checkpoint_file = os.path.join(self.save_dir, 'checkpoints.json')

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoints = json.load(f)
        else:
            checkpoints = []

        checkpoints.append(checkpoint_info)

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoints, f, indent=2)


class ExperimentTracker:
    """Track and compare multiple training runs."""

    def __init__(self, experiment_name: str, base_dir: str = 'runs'):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.agents = {}
        self.start_time = time.time()

    def add_agent(self, agent_name: str) -> TrainingLogger:
        """Add a new agent to track."""
        agent_dir = os.path.join(self.experiment_dir, agent_name)
        logger = TrainingLogger(agent_name, agent_dir)
        self.agents[agent_name] = logger
        return logger

    def compare_agents(self, save_path: Optional[str] = None):
        """Create comparison plots for all tracked agents."""
        if save_path is None:
            save_path = os.path.join(self.experiment_dir, 'agent_comparison.png')

        # Filter out agents with no data
        agents_with_data = {name: logger for name, logger in self.agents.items()
                           if len(logger.episode_rewards) > 0}

        if not agents_with_data:
            print("No agents with training data to compare.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Agent Comparison: {self.experiment_name}', fontsize=16)

        for agent_name, logger in agents_with_data.items():
            episodes = list(range(1, len(logger.episode_rewards) + 1))

            # Moving average rewards
            axes[0, 0].plot(episodes, logger.moving_avg_rewards,
                           label=agent_name, linewidth=2)

            # Episode progress
            axes[0, 1].plot(episodes, logger.episode_progress,
                           label=agent_name, alpha=0.7)

            # Learning efficiency (reward per step)
            cumulative_steps = np.cumsum(logger.episode_lengths)
            axes[1, 0].plot(cumulative_steps, logger.moving_avg_rewards,
                           label=agent_name, linewidth=2)

        # Configure subplots
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].set_title('Learning Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Progress (0-1)')
        axes[0, 1].set_title('Track Progress')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_xlabel('Total Steps')
        axes[1, 0].set_ylabel('Average Reward')
        axes[1, 0].set_title('Sample Efficiency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Summary statistics table
        axes[1, 1].axis('off')
        stats_rows = [['Agent', 'Best Reward', 'Final Avg', 'Episodes']]
        for agent_name, logger in agents_with_data.items():
            stats_rows.append([
                agent_name,
                f"{logger.best_reward:.2f}",
                f"{logger.moving_avg_rewards[-1]:.2f}",
                f"{len(logger.episode_rewards)}"
            ])

        table = axes[1, 1].table(cellText=stats_rows, loc='center',
                                cellLoc='left', colWidths=[0.3, 0.25, 0.25, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(len(stats_rows[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Agent comparison plot saved to {save_path}")

    def save_experiment_summary(self):
        """Save complete experiment summary."""
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time,
            'total_time': time.time() - self.start_time,
            'agents': {}
        }

        for agent_name, logger in self.agents.items():
            if len(logger.episode_rewards) > 0:
                summary['agents'][agent_name] = {
                    'total_episodes': len(logger.episode_rewards),
                    'total_steps': logger.total_steps,
                    'best_reward': float(logger.best_reward),
                    'final_avg_reward': float(logger.moving_avg_rewards[-1]),
                    'overall_avg_reward': float(np.mean(logger.episode_rewards))
                }
            else:
                summary['agents'][agent_name] = {
                    'total_episodes': 0,
                    'total_steps': 0,
                    'best_reward': None,
                    'final_avg_reward': None,
                    'overall_avg_reward': None,
                    'status': 'No training data'
                }

        summary_file = os.path.join(self.experiment_dir, 'experiment_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Experiment summary saved to {summary_file}")

"""
Visualization tools for analyzing training results.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import glob


def load_training_stats(model_path: str) -> Dict:
    """Load training statistics from saved model."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        return checkpoint.get('train_stats', {})
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return {}


def plot_loss_curves(model_paths: List[str], labels: List[str] = None):
    """
    Plot training loss curves for multiple models.
    
    Args:
        model_paths: List of paths to saved models
        labels: Optional labels for each model
    """
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(model_paths))]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for model_path, label in zip(model_paths, labels):
        stats = load_training_stats(model_path)
        
        if not stats:
            continue
        
        # Policy loss
        if 'policy_loss' in stats:
            axes[0, 0].plot(stats['policy_loss'], label=label, alpha=0.7)
        
        # Value loss
        if 'value_loss' in stats:
            axes[0, 1].plot(stats['value_loss'], label=label, alpha=0.7)
        
        # Entropy
        if 'entropy' in stats:
            axes[1, 0].plot(stats['entropy'], label=label, alpha=0.7)
        
        # Total loss
        if 'total_loss' in stats:
            axes[1, 1].plot(stats['total_loss'], label=label, alpha=0.7)
    
    axes[0, 0].set_title('Policy Loss')
    axes[0, 0].set_xlabel('Update Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Value Loss')
    axes[0, 1].set_xlabel('Update Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Entropy')
    axes[1, 0].set_xlabel('Update Step')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Total Loss')
    axes[1, 1].set_xlabel('Update Step')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loss_comparison.png', dpi=150)
    plt.show()
    print("Loss comparison saved to 'loss_comparison.png'")


def compare_checkpoints(save_dir: str):
    """
    Compare all checkpoints in a directory.
    
    Args:
        save_dir: Directory containing checkpoint files
    """
    # Find all checkpoint files
    checkpoint_files = sorted(glob.glob(os.path.join(save_dir, 'checkpoint_*.pt')))
    
    if not checkpoint_files:
        print(f"No checkpoints found in {save_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    # Extract episode numbers and labels
    labels = []
    for path in checkpoint_files:
        filename = os.path.basename(path)
        ep_num = filename.split('_ep')[1].split('.')[0]
        labels.append(f"Episode {ep_num}")
    
    plot_loss_curves(checkpoint_files, labels)


def analyze_model_performance(model_path: str):
    """
    Detailed analysis of a single model's training.
    
    Args:
        model_path: Path to saved model
    """
    stats = load_training_stats(model_path)
    
    if not stats:
        print("No statistics found in model")
        return
    
    print("=" * 60)
    print(f"Model Analysis: {model_path}")
    print("=" * 60)
    
    for key, values in stats.items():
        if values:
            print(f"\n{key.upper()}:")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Std:  {np.std(values):.4f}")
            print(f"  Min:  {np.min(values):.4f}")
            print(f"  Max:  {np.max(values):.4f}")
            print(f"  Final: {values[-1]:.4f}")
    
    print("=" * 60)
    
    # Plot individual model statistics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    if 'policy_loss' in stats:
        axes[0, 0].plot(stats['policy_loss'])
        axes[0, 0].set_title('Policy Loss Over Training')
        axes[0, 0].set_xlabel('Update Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    if 'value_loss' in stats:
        axes[0, 1].plot(stats['value_loss'])
        axes[0, 1].set_title('Value Loss Over Training')
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    if 'entropy' in stats:
        axes[1, 0].plot(stats['entropy'])
        axes[1, 0].set_title('Entropy Over Training')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].grid(True, alpha=0.3)
    
    if 'total_loss' in stats:
        axes[1, 1].plot(stats['total_loss'])
        axes[1, 1].set_title('Total Loss Over Training')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with model name
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    save_path = f'{model_name}_analysis.png'
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Analysis plot saved to '{save_path}'")


def watch_trained_agent(model_path: str, num_episodes: int = 5):
    """
    Watch a trained agent play (simplified version).
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of episodes to watch
    """
    from train import evaluate_agent
    
    print(f"Watching agent from {model_path}...")
    evaluate_agent(model_path, num_episodes=num_episodes, render=True)


def main():
    """Interactive menu for visualization tools."""
    print("=" * 60)
    print("Training Results Visualization Tool")
    print("=" * 60)
    print("1. Analyze single model")
    print("2. Compare all checkpoints in directory")
    print("3. Watch trained agent")
    print("4. Compare specific models")
    print("=" * 60)
    
    choice = input("Select option (1-4): ").strip()
    
    if choice == "1":
        model_path = input("Enter model path: ").strip()
        if os.path.exists(model_path):
            analyze_model_performance(model_path)
        else:
            print(f"File not found: {model_path}")
    
    elif choice == "2":
        save_dir = input("Enter directory path (default: models): ").strip()
        if not save_dir:
            save_dir = "models"
        if os.path.exists(save_dir):
            compare_checkpoints(save_dir)
        else:
            print(f"Directory not found: {save_dir}")
    
    elif choice == "3":
        model_path = input("Enter model path: ").strip()
        if os.path.exists(model_path):
            num_eps = input("Number of episodes (default: 5): ").strip()
            num_eps = int(num_eps) if num_eps else 5
            watch_trained_agent(model_path, num_eps)
        else:
            print(f"File not found: {model_path}")
    
    elif choice == "4":
        print("Enter model paths (one per line, empty line to finish):")
        paths = []
        while True:
            path = input().strip()
            if not path:
                break
            if os.path.exists(path):
                paths.append(path)
            else:
                print(f"Warning: {path} not found, skipping")
        
        if paths:
            labels = [f"Model {i+1}" for i in range(len(paths))]
            plot_loss_curves(paths, labels)
        else:
            print("No valid paths provided")
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
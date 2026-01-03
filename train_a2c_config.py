"""
Training script that uses configuration presets.
"""
import argparse
from config import get_config, print_config
from train import train_a2c, evaluate_agent


def main():
    parser = argparse.ArgumentParser(description='Train A2C agent with configuration preset')
    parser.add_argument(
        '--config',
        type=str,
        default='base',
        choices=['base', 'quick', 'intensive', 'fast', 'stable'],
        help='Configuration preset to use'
    )
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Evaluate model after training'
    )
    parser.add_argument(
        '--eval-only',
        type=str,
        default=None,
        help='Only evaluate the specified model (skip training)'
    )
    parser.add_argument(
        '--num-eval-episodes',
        type=int,
        default=10,
        help='Number of episodes for evaluation'
    )
    
    args = parser.parse_args()
    
    # Evaluation only mode
    if args.eval_only:
        print(f"Evaluating model: {args.eval_only}")
        evaluate_agent(
            args.eval_only,
            num_episodes=args.num_eval_episodes,
            render=True
        )
        return
    
    # Get configuration
    config = get_config(args.config)
    print_config(config)
    
    # Confirm training
    response = input("\nStart training with this configuration? (y/n): ").strip().lower()
    if response != 'y':
        print("Training cancelled.")
        return
    
    # Train agent
    print("\nStarting training...")
    train_a2c(**config)
    
    # Evaluate if requested
    if args.eval:
        import os
        best_model_path = os.path.join(config['save_dir'], 'best_model.pt')
        
        if os.path.exists(best_model_path):
            print("\n" + "=" * 60)
            print("Evaluating best model...")
            print("=" * 60)
            evaluate_agent(
                best_model_path,
                num_episodes=args.num_eval_episodes,
                render=True
            )
        else:
            print(f"\nBest model not found at {best_model_path}")


if __name__ == "__main__":
    main()
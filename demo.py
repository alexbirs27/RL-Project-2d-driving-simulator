"""
Quick demo script to test the environment and agent.
"""
import numpy as np
from environment import RacingEnv
from agent import A2CAgent


def test_environment():
    """Test the environment with random actions."""
    print("Testing environment with random actions...")
    env = RacingEnv(render_mode='human', max_steps=500)
    
    state, _ = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial state shape: {state.shape}")
    
    total_reward = 0
    done = False
    step = 0
    
    while not done:
        # Random action
        action = env.action_space.sample()
        
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        step += 1
        
        env.render()
        
        if step % 100 == 0:
            print(f"Step {step}: Reward = {reward:.2f}, "
                  f"Progress = {info['progress']:.2f}, "
                  f"Velocity = {info['velocity']:.2f}")
    
    print(f"\nEpisode finished!")
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final progress: {info['progress']:.2f}")
    
    env.close()


def test_agent_untrained():
    """Test an untrained agent."""
    print("\nTesting untrained agent...")
    env = RacingEnv(render_mode='human', max_steps=500)
    agent = A2CAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    state, _ = env.reset()
    total_reward = 0
    done = False
    step = 0
    
    while not done:
        # Agent selects action
        action, log_prob, value = agent.select_action(state)
        
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        step += 1
        
        env.render()
        
        if step % 100 == 0:
            print(f"Step {step}: Action = {action}, Value = {value:.2f}, "
                  f"Progress = {info['progress']:.2f}")
    
    print(f"\nEpisode finished!")
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.2f}")
    
    env.close()


def quick_train():
    """Quick training demonstration (just 50 episodes)."""
    from train import train_a2c
    
    print("\nQuick training demo (50 episodes)...")
    train_a2c(
        num_episodes=50,
        max_steps_per_episode=1000,
        update_frequency=5,
        save_frequency=25,
        render_frequency=10,
        learning_rate=3e-4,
        gamma=0.99,
        save_dir='demo_models'
    )


if __name__ == "__main__":
    # Choose which demo to run
    print("=" * 60)
    print("Racing RL Demo")
    print("=" * 60)
    print("1. Test environment with random actions")
    print("2. Test untrained agent")
    print("3. Quick training demo (50 episodes)")
    print("=" * 60)
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        test_environment()
    elif choice == "2":
        test_agent_untrained()
    elif choice == "3":
        quick_train()
    else:
        print("Invalid choice. Running environment test...")
        test_environment()
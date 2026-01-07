import numpy as np                      # for easy work with vectors/matrices and random/shuffle
import torch                            # framework for neural networks and tensor operations
import matplotlib.pyplot as plt

from env import make_env
from .agent import PPOAgent


# Main Training Loop (collection + update)
def train(render: bool = False):
    render_mode = "human" if render else None
    env = make_env(render_mode=render_mode)  # DrivingEnv: 8 observations, 5 actions
            
    agent = PPOAgent()

    epochs = 400  # Increased for better learning
    steps_per_epoch = agent.steps_per_epoch
    reward_history = []
    laps_per_epoch = []                     # Track lap completions per epoch
    best_reward = -float('inf')  # Track best reward for checkpointing

    for epoch in range(epochs):
        # Update learning rate (linear decay from initial_lr to 10% of initial)
        agent.update_learning_rate(epoch, epochs)
        # Update entropy coefficient (decay for less exploration over time)
        agent.update_entropy_coef(epoch, epochs)

        observations = []
        actions = []
        logps = []                          # Log-probabilities of actions under the old policy
        values = []                         # V(s) values estimated by the critic
        rewards = []
        dones = []                          # Whether the episode ended at that step

        state, _ = env.reset()              # Reset environment - receive initial state
        ep_reward = 0                       # Accumulated reward for the current episode
        epoch_laps = 0                      # Count lap completions this epoch

        for step in range(steps_per_epoch):

            action, logp, value = agent.act(state)       # Agent chooses an action + logp + value for the current state
            next_state, reward, terminated, truncated, _ = env.step(action)  # Apply action in env
            done = terminated or truncated               # Episode is done if terminated or truncated

            observations.append(state)
            actions.append(action)
            logps.append(logp)
            values.append(value)
            rewards.append(reward)
            dones.append(done)

            ep_reward += reward
            state = next_state

            if render:
                env.render()

            if terminated:                              # Lap completed (not just truncated)
                epoch_laps += 1

            if done:
                reward_history.append(ep_reward)
                state, _ = env.reset()                   # Reset for a new episode
                ep_reward = 0                            # Reset ep_reward

        laps_per_epoch.append(epoch_laps)

        # Convert to tensors (prepare data for PyTorch training)
        obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32)  # Observations -> float32 tensor
        actions_tensor = torch.tensor(actions)                                   # Actions -> tensor (int)
        logp_tensor = torch.stack(logps)                                         # Logps are tensors -> stack them into one tensor
        values_list = [v.item() for v in values]                                 # Values -> list of floats (extracted from tensor)
        next_value = agent.value_fn(torch.tensor(state, dtype=torch.float32)).item()  # Value of the last state (bootstrap) as float

        advantages, returns = agent.compute_gae(rewards, values_list, next_value, dones)  # Calculate advantages and returns

        # Train PPO (update actor + critic)
        agent.train(obs_tensor, actions_tensor, logp_tensor, advantages, returns)  # Perform PPO update on collected data

        if len(reward_history) >= 10:
            mean_reward = np.mean(reward_history[-10:])
        else:
            mean_reward = np.mean(reward_history)

        print(f"[Epoch {epoch}] Mean Reward (last 10 episodes): {mean_reward:.2f}")

        # Save checkpoint if this is the best model so far
        if mean_reward > best_reward:
            best_reward = mean_reward
            agent.save_checkpoint('best_model.pt', epoch, mean_reward)
            print(f"  -> New best model saved! (reward: {mean_reward:.2f})")

    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    # Top plot: Episode rewards
    ax1.plot(reward_history, label="Episode Reward", alpha=0.4, color='tab:blue')
    if len(reward_history) > 20:
        moving_avg = np.convolve(reward_history, np.ones(20) / 20, mode="valid")
        ax1.plot(moving_avg, label="Moving Average (20)", linewidth=2, color='tab:orange')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("PPO Training Performance")
    ax1.legend()
    ax1.grid(True)

    # Bottom plot: Laps completed per epoch
    ax2.bar(range(len(laps_per_epoch)), laps_per_epoch, alpha=0.6, color='tab:green', label="Laps per Epoch")
    if len(laps_per_epoch) > 10:
        laps_moving_avg = np.convolve(laps_per_epoch, np.ones(10) / 10, mode="valid")
        ax2.plot(range(9, len(laps_per_epoch)), laps_moving_avg, label="Moving Average (10)",
                 linewidth=2, color='tab:red')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Laps Completed")
    ax2.set_title("Lap Completions per Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("ppo_training_rewards.png")
    plt.show()

    # Save final model
    final_reward = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else np.mean(reward_history)
    agent.save_checkpoint('final_model.pt', epochs - 1, final_reward)
    print(f"\nTraining complete!")
    print(f"  Best model reward: {best_reward:.2f}")
    print(f"  Final model reward: {final_reward:.2f}")

    return agent, reward_history, laps_per_epoch    


if __name__ == "__main__":
    import sys
    render = "--render" in sys.argv
    train(render=render)

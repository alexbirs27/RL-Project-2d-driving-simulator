import sys
import os
import numpy as np                      # for easy work with vectors/matrices and random/shuffle
import torch                            # framework for neural networks and tensor operations
import matplotlib.pyplot as plt
from datetime import datetime

# Setup path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.env.driving_env import make_env
from .agent import PPOAgent


# Main Training Loop (collection + update)
def train(render: bool = False):
    # Generate unique timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"ppo_run_{timestamp}"

    print(f"--- RULARE NOUA: {run_name} ---")

    # Setup folders
    models_dir = os.path.join(project_root, 'models')
    plots_dir = os.path.join(project_root, 'plots')
    logs_dir = os.path.join(project_root, 'logs')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Log file
    log_file = os.path.join(logs_dir, f"{run_name}_log.csv")
    with open(log_file, "w") as f:
        f.write("epoch,mean_reward\n")

    # Model paths
    best_model_path = os.path.join(models_dir, f'{run_name}_BEST_model.pt')
    final_model_path = os.path.join(models_dir, f'{run_name}_FINAL_model.pt')

    render_mode = "human" if render else None
    env = make_env(render_mode=render_mode)  # DrivingEnv: 8 observations, 5 actions

    agent = PPOAgent()

    epochs = 200
    steps_per_epoch = agent.steps_per_epoch
    reward_history = []
    best_reward = -float('inf')

    for epoch in range(epochs):

        observations = []
        actions = []
        logps = []                          # Log-probabilities of actions under the old policy
        values = []                         # V(s) values estimated by the critic
        rewards = []
        dones = []                          # Whether the episode ended at that step

        state, _ = env.reset()              # Reset environment - receive initial state
        ep_reward = 0                       # Accumulated reward for the current episode

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

            if done:
                reward_history.append(ep_reward)
                state, _ = env.reset()                   # Reset for a new episode
                ep_reward = 0                            # Reset ep_reward

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
            mean_reward = np.mean(reward_history) if reward_history else 0.0

        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            agent.save(best_model_path)
            print(f"!!! RECORD NOU: {best_reward:.2f} -> Model salvat.")

        print(f"[Epoch {epoch}] Mean Reward (last 10 episodes): {mean_reward:.2f} | Best: {best_reward:.2f}")

        # Log to CSV
        with open(log_file, "a") as f:
            f.write(f"{epoch},{mean_reward}\n")

    # Save final model
    agent.save(final_model_path)
    print(f"Model final salvat in: {final_model_path}")
    print(f"Best Reward: {best_reward:.2f}")

    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, label="Episode Reward", alpha=0.4)

    if len(reward_history) > 20:
        moving_avg = np.convolve(reward_history, np.ones(20) / 20, mode="valid")
        plt.plot(moving_avg, label="Moving Average (20)", linewidth=2)

    # Linie pentru best reward
    plt.axhline(y=best_reward, color='red', linestyle=':', label=f'Best Reward ({best_reward:.0f})')

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"PPO Training: {run_name}")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(plots_dir, f'{run_name}_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Grafic salvat in: {plot_path}")

    env.close()
    return agent, reward_history    


if __name__ == "__main__":
    import sys
    render = "--render" in sys.argv
    train(render=render)

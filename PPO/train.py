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

    epochs = 200
    steps_per_epoch = agent.steps_per_epoch
    reward_history = []

    for epoch in range(epochs):
        # Update learning rate (linear decay from initial_lr to 0)
        agent.update_learning_rate(epoch, epochs)

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
            mean_reward = np.mean(reward_history)

        print(f"[Epoch {epoch}] Mean Reward (last 10 episodes): {mean_reward:.2f}")

    
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label="Episode Reward", alpha=0.4)

    if len(reward_history) > 20:
        moving_avg = np.convolve(reward_history, np.ones(20) / 20, mode="valid")
        plt.plot(moving_avg, label="Moving Average (20)", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Performance")
    plt.legend()
    plt.grid(True)

    plt.savefig("ppo_training_rewards.png")
    plt.show()

    return agent, reward_history    


if __name__ == "__main__":
    import sys
    render = "--render" in sys.argv
    train(render=render)

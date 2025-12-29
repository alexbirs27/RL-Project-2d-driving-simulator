import numpy as np                      # for easy work with vectors/matrices and random/shuffle
import torch                            # framework for neural networks and tensor operations

from env import make_env
from .agent import PPOAgent


# Main Training Loop (collection + update)
def train():
    env = make_env()  # DrivingEnv: 8 observations, 5 actions

    state_dim = env.observation_space.shape[0]  # 8
    action_dim = env.action_space.n             # 5
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)

    epochs = 200
    steps_per_epoch = agent.steps_per_epoch
    reward_history = []

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

    # TODO: Add logging/plotting of reward_history
    # TODO: Add model saving/checkpointing
    return agent, reward_history


if __name__ == "__main__":
    train()

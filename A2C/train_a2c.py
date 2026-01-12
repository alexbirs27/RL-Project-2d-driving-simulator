import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pygame
from datetime import datetime

# Setup cai import
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Config si module
from config import *
from env import make_env
from A2C.agent import A2CAgent

# --- GENERARE TIMESTAMP UNIC ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"a2c_run_{timestamp}"

print(f"--- RULARE NOUA: {run_name} ---")

# --- SETUP FOLDERE ---
logs_dir = os.path.join(project_root, 'logs')
models_dir = os.path.join(project_root, 'models')
plots_dir = os.path.join(project_root, 'plots')

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Initializare fisier log CSV
log_file = os.path.join(logs_dir, f"{run_name}_log.csv")
with open(log_file, "w") as f:
    f.write("episode,reward,loss\n")
    
# Calea pentru CEL MAI BUN model
best_model_path = os.path.join(models_dir, f'{run_name}_BEST_model.pt')
# ---------------------

def train_a2c_script(render=False, render_freq=1):
    """
    Train A2C agent.

    Args:
        render: Enable rendering
        render_freq: Render every N episodes (e.g., 5 = render 1 out of 5 episodes)
    """
    # Initializare mediu
    env = make_env(render_mode=None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Initializare A2C (state_dim={state_dim}, action_dim={action_dim}, LR={A2C_LEARNING_RATE})...")
    agent = A2CAgent(state_dim=state_dim, action_dim=action_dim, lr=A2C_LEARNING_RATE)

    num_episodes = A2C_TOTAL_EPISODES

    history_rewards = []
    history_loss = []

    # --- VARIABILA PENTRU SALVAREA CELUI MAI BUN MODEL ---
    best_reward = -float('inf')

    print(f"--- Incepe Antrenamentul A2C ({num_episodes} episoade) ---")

    for episode in range(num_episodes):
        # Enable rendering for this episode if it's a render episode
        should_render = render and (episode % render_freq == 0)
        if should_render:
            env.close()
            env = make_env(render_mode="human")

        state, _ = env.reset()
        done = False

        rewards = []
        log_probs = []
        states = []
        dones = []

        total_reward = 0

        while not done:
            # Handle pygame events to prevent window freeze
            if should_render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nTraining interrupted by user")
                        env.close()
                        return

            action, log_prob = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)

            total_reward += reward
            state = next_state

            if should_render:
                env.render()

        # Disable rendering after episode if it was enabled
        if should_render and episode < num_episodes - 1:
            env.close()
            env = make_env(render_mode=None)

        loss = agent.update(rewards, log_probs, states, dones, next_state)
        
        history_rewards.append(total_reward)
        history_loss.append(loss)
        
        # --- SALVARE CHECKPOINT INTELIGENT ---
        # Salvam modelul doar daca a doborat recordul
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(best_model_path)
            print(f"!!! RECORD NOU: {best_reward:.2f} -> Model salvat in {os.path.basename(best_model_path)}")
        # -------------------------------------
        
        if (episode + 1) % 10 == 0:
            print(f"Episod {episode+1}/{num_episodes} | Reward: {total_reward:.2f} | Loss: {loss:.4f} | Best: {best_reward:.2f}")

        with open(log_file, "a") as f:
            f.write(f"{episode+1},{total_reward},{loss}\n")

    # Salvare model final (chiar daca e mai prost, e bine sa il avem)
    final_path = os.path.join(models_dir, f'{run_name}_FINAL_model.pt')
    agent.save(final_path)
    print(f"Antrenament terminat. Best Reward: {best_reward:.2f}")
    
    # --- GENERARE GRAFICE ---
    print("Generare grafice...")
    
    window_size = 20
    if len(history_rewards) >= window_size:
        moving_avg = np.convolve(history_rewards, np.ones(window_size)/window_size, mode='valid')
    else:
        moving_avg = history_rewards

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(history_rewards, label='Episode Reward', alpha=0.3, color='blue')
    ax1.plot(range(len(moving_avg)), moving_avg, label=f'Moving Avg ({window_size})', color='red', linewidth=2)
    # Adaugam o linie orizontala pentru cel mai bun scor
    ax1.axhline(y=best_reward, color='green', linestyle='--', label=f'Best ({best_reward:.0f})')
    
    ax1.set_title(f'A2C Rewards: {run_name}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history_loss, label='Training Loss', color='orange')
    ax2.set_title(f'A2C Loss: {run_name}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'{run_name}_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Grafic salvat in: {plot_path}")
    
    env.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train A2C agent")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--render-freq", type=int, default=1, help="Render every N episodes (default: 1)")
    args = parser.parse_args()

    train_a2c_script(render=args.render, render_freq=args.render_freq)
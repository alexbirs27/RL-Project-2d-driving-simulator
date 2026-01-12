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
from ARS.agent import ARSAgent

# --- GENERARE TIMESTAMP UNIC ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"ars_run_{timestamp}"

print(f"--- RULARE NOUA: {run_name} ---")

# --- SETUP FOLDERE ---
logs_dir = os.path.join(project_root, 'logs')
models_dir = os.path.join(project_root, 'models')
plots_dir = os.path.join(project_root, 'plots')

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Fisier Log UNIC
log_file = os.path.join(logs_dir, f"{run_name}_log.csv")
with open(log_file, "w") as f:
    f.write("iteration,avg_reward,max_reward\n")

# Calea pentru CEL MAI BUN model
best_model_path = os.path.join(models_dir, f'{run_name}_BEST_weights.npy')
# ---------------------

def train_ars_script(render=False, render_freq=1):
    """
    Train ARS agent.

    Args:
        render: Enable rendering
        render_freq: Render every N iterations (e.g., 5 = render 1 out of 5 iterations)
    """
    # Initializare mediu
    env = make_env(render_mode=None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initializare Agent cu parametri din Config
    print(f"Initializare ARS (state_dim={state_dim}, action_dim={action_dim}, StepSize={ARS_LEARNING_RATE})...")
    agent = ARSAgent(state_dim=state_dim, action_dim=action_dim, learning_rate=ARS_LEARNING_RATE)

    # Numarul de iteratii din Config
    num_iterations = ARS_TOTAL_ITERATIONS

    history_avg_reward = []
    history_max_reward = []

    # Variabila pentru record
    best_reward = -float('inf')

    print(f"--- Incepe Antrenamentul ARS ({num_iterations} iteratii) ---")

    for i in range(num_iterations):
        # Enable rendering for this iteration if it's a render iteration
        should_render = render and (i % render_freq == 0)
        if should_render:
            env.close()
            env = make_env(render_mode="human")

        # 1. Generare perturbari (zgomot)
        deltas = [np.random.randn(*agent.weights.shape) for _ in range(agent.num_deltas)]
        rollouts = []
        rewards_list = []

        # 2. Evaluare fiecare perturbare (+ si -)
        for delta in deltas:
            r_pos = run_episode(env, agent, delta, direction="plus", should_render=should_render)
            r_neg = run_episode(env, agent, delta, direction="minus", should_render=should_render)

            rollouts.append((r_pos, r_neg, delta))
            rewards_list.extend([r_pos, r_neg])

        # Disable rendering after iteration if it was enabled
        if should_render and i < num_iterations - 1:
            env.close()
            env = make_env(render_mode=None)

        # 3. Update greutati agent
        sigma_r = np.std(rewards_list) if np.std(rewards_list) > 0 else 1
        agent.update(rollouts, sigma_r)
        
        # Statistici
        avg_reward = np.mean(rewards_list)
        max_reward = np.max(rewards_list)
        
        history_avg_reward.append(avg_reward)
        history_max_reward.append(max_reward)
        
        # --- SALVARE CHECKPOINT INTELIGENT ---
        # La ARS ne intereseaza daca am gasit o directie care duce la un scor maxim
        if max_reward > best_reward:
            best_reward = max_reward
            np.save(best_model_path, agent.weights)
            print(f"!!! RECORD NOU: {best_reward:.2f} -> Model salvat.")
        # -------------------------------------
        
        print(f"Iteratia {i+1}/{num_iterations} | Avg: {avg_reward:.2f} | Max: {max_reward:.2f} | Best: {best_reward:.2f}")

        with open(log_file, "a") as f:
            f.write(f"{i+1},{avg_reward},{max_reward}\n")

    # --- SALVARE MODEL FINAL ---
    final_path = os.path.join(models_dir, f'{run_name}_FINAL_weights.npy')
    np.save(final_path, agent.weights)
    print(f"Antrenament terminat. Best Reward: {best_reward:.2f}")
    print(f"Model final salvat in: {final_path}")
    
    # --- GENERARE GRAFIC UNIC ---
    print("Generare grafic...")
    plt.figure(figsize=(10, 6))
    
    plt.plot(history_avg_reward, label='Average Reward', color='blue', alpha=0.7)
    plt.plot(history_max_reward, label='Max Reward', color='green', linestyle='--')
    
    # Linie orizontala pentru record
    plt.axhline(y=best_reward, color='red', linestyle=':', label=f'Best Ever ({best_reward:.0f})')

    plt.title(f'ARS Training: {run_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(plots_dir, f'{run_name}_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Grafic salvat in: {plot_path}")
    
    env.close()

def run_episode(env, agent, delta, direction, should_render=False):
    state, _ = env.reset()
    total_reward = 0
    done = False

    # Limita de siguranta pentru bucle infinite (optional)
    steps = 0
    max_steps = 2000

    while not done and steps < max_steps:
        # Handle pygame events to prevent window freeze
        if should_render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\nTraining interrupted by user")
                    env.close()
                    return total_reward

        action = agent.select_action(state, delta, direction)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        if should_render:
            env.render()

    return total_reward

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ARS agent")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--render-freq", type=int, default=1, help="Render every N iterations (default: 1)")
    args = parser.parse_args()

    train_ars_script(render=args.render, render_freq=args.render_freq)

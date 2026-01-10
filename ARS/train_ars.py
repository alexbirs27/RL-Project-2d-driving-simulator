import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Setup path to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from project root
from config import (
    OBS_DIM, ACTION_DIM,
    ARS_LEARNING_RATE, ARS_NOISE, ARS_NUM_DELTAS, ARS_TOTAL_ITERATIONS
)
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

def train_ars_script():
    # Initializare mediu
    env = make_env(render_mode=None)

    # Use dimensions from config
    state_dim = OBS_DIM
    action_dim = ACTION_DIM

    print(f"Initializare ARS (state_dim={state_dim}, action_dim={action_dim}, StepSize={ARS_LEARNING_RATE})...")
    agent = ARSAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=ARS_LEARNING_RATE,
        noise=ARS_NOISE,
        num_deltas=ARS_NUM_DELTAS
    )

    num_iterations = ARS_TOTAL_ITERATIONS

    history_avg_reward = []
    history_max_reward = []

    # Variabila pentru record
    best_reward = -float('inf')

    print(f"--- Incepe Antrenamentul ARS ({num_iterations} iteratii) ---")

    for i in range(num_iterations):
        # 1. Generare perturbari (zgomot)
        deltas = [np.random.randn(*agent.weights.shape) for _ in range(agent.num_deltas)]
        rollouts = []
        rewards_list = []

        # 2. Evaluare fiecare perturbare (+ si -)
        for delta in deltas:
            r_pos = run_episode(env, agent, delta, direction="plus")
            r_neg = run_episode(env, agent, delta, direction="minus")

            rollouts.append((r_pos, r_neg, delta))
            rewards_list.extend([r_pos, r_neg])

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

def run_episode(env, agent, delta, direction):
    state, _ = env.reset()
    total_reward = 0
    done = False

    # Limita de siguranta pentru bucle infinite (optional)
    steps = 0
    max_steps = 2000

    while not done and steps < max_steps:
        action = agent.select_action(state, delta, direction)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    return total_reward

if __name__ == "__main__":
    train_ars_script()

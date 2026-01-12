import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Setup cai import
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import module
from env import make_env
from src.agents.ars.agent import ARSAgent

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
    f.write("iteration,avg_reward,max_reward,lap_completions,lr,noise\n")

# Calea pentru CEL MAI BUN model
best_model_path = os.path.join(models_dir, f'{run_name}_BEST_weights.npz')
# ---------------------


def run_episode(env, agent, delta, direction, collect_states=False):
    """
    Run a single episode with the given perturbation.

    Args:
        env: Environment
        agent: ARS agent
        delta: Perturbation matrix
        direction: "plus" or "minus"
        collect_states: If True, also return collected states for normalization

    Returns:
        total_reward, lap_completed, (states if collect_states)
    """
    state, _ = env.reset()
    total_reward = 0
    done = False
    lap_completed = False
    states = []

    steps = 0
    max_steps = 3000

    while not done and steps < max_steps:
        if collect_states:
            states.append(state.copy())

        action = agent.select_action(state, delta, direction)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

        if info.get('lap_complete', False):
            lap_completed = True

    if collect_states:
        return total_reward, lap_completed, states
    return total_reward, lap_completed


def train_ars_script():
    # Initializare mediu
    env = make_env(render_mode=None)

    # Get dimensions from environment (dynamic)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Hyperparameters - ARS V2
    num_iterations = 500  # More iterations for better convergence
    num_deltas = 32  # More deltas for better exploration
    num_best_deltas = 16  # Top-k selection
    learning_rate = 0.03  # Higher initial LR
    noise = 0.025  # Initial noise level

    print(f"Initializare ARS-V2 (state_dim={state_dim}, action_dim={action_dim})...")
    print(f"  LR={learning_rate}, noise={noise}, deltas={num_deltas}")
    agent = ARSAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        noise=noise,
        num_deltas=num_deltas,
        num_best_deltas=num_best_deltas,
        normalize_states=True
    )

    history_avg_reward = []
    history_max_reward = []
    history_lap_completions = []

    # Variabila pentru record
    best_reward = -float('inf')
    best_mean_reward = -float('inf')
    total_laps = 0

    print(f"--- Incepe Antrenamentul ARS-V2 ({num_iterations} iteratii) ---")
    print(f"    Deltas: {num_deltas}, Top-k: {num_best_deltas}")
    print(f"    State normalization: enabled")

    # Warmup: collect states for normalization
    print("Warmup: collecting states for normalization...")
    warmup_states = []
    for _ in range(5):
        delta = np.random.randn(*agent.weights.shape)
        _, _, states = run_episode(env, agent, delta, "plus", collect_states=True)
        warmup_states.extend(states)
    agent.update_state_stats(warmup_states)
    print(f"  Collected {len(warmup_states)} states for normalization")

    for i in range(num_iterations):
        # Decay hyperparameters
        agent.decay_hyperparameters(i, num_iterations)

        # 1. Generare perturbari (zgomot)
        deltas = [np.random.randn(*agent.weights.shape) for _ in range(agent.num_deltas)]
        rollouts = []
        rewards_list = []
        iteration_laps = 0
        all_states = []

        # 2. Evaluare fiecare perturbare (+ si -)
        for delta in deltas:
            r_pos, lap_pos, states_pos = run_episode(env, agent, delta, "plus", collect_states=True)
            r_neg, lap_neg, states_neg = run_episode(env, agent, delta, "minus", collect_states=True)

            rollouts.append((r_pos, r_neg, delta))
            rewards_list.extend([r_pos, r_neg])

            if lap_pos:
                iteration_laps += 1
            if lap_neg:
                iteration_laps += 1

            all_states.extend(states_pos)
            all_states.extend(states_neg)

        # Update state normalization
        if all_states:
            agent.update_state_stats(all_states)

        # 3. Update greutati agent
        sigma_r = np.std(rewards_list) if np.std(rewards_list) > 0 else 1
        agent.update(rollouts, sigma_r)

        # Statistici
        avg_reward = np.mean(rewards_list)
        max_reward = np.max(rewards_list)
        total_laps += iteration_laps

        history_avg_reward.append(avg_reward)
        history_max_reward.append(max_reward)
        history_lap_completions.append(iteration_laps)

        # Calculate mean over last 20 iterations
        mean_reward = np.mean(history_avg_reward[-20:]) if len(history_avg_reward) >= 20 else np.mean(history_avg_reward)

        # --- SALVARE CHECKPOINT INTELIGENT ---
        if max_reward > best_reward:
            best_reward = max_reward
            agent.save(best_model_path)
            print(f"!!! RECORD NOU (max): {best_reward:.2f} -> Model salvat.")

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
        # -------------------------------------

        # Print progress
        print(
            f"Iter {i+1}/{num_iterations} | "
            f"Avg: {avg_reward:.1f} | "
            f"Max: {max_reward:.1f} | "
            f"Best: {best_reward:.1f} | "
            f"Laps: {iteration_laps} | "
            f"Total Laps: {total_laps} | "
            f"LR: {agent.lr:.4f} | "
            f"Noise: {agent.noise:.4f}"
        )

        with open(log_file, "a") as f:
            f.write(f"{i+1},{avg_reward},{max_reward},{iteration_laps},{agent.lr},{agent.noise}\n")

    # --- SALVARE MODEL FINAL ---
    final_path = os.path.join(models_dir, f'{run_name}_FINAL_weights.npz')
    agent.save(final_path)
    print(f"\nAntrenament terminat!")
    print(f"Best Single Reward: {best_reward:.2f}")
    print(f"Best Mean Reward (20 iter): {best_mean_reward:.2f}")
    print(f"Total Laps Completed: {total_laps}")
    print(f"Model salvat in: {final_path}")

    # --- GENERARE GRAFICE ---
    print("Generare grafice...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Rewards
    ax1 = axes[0, 0]
    ax1.plot(history_avg_reward, label='Average Reward', color='blue', alpha=0.7)
    ax1.plot(history_max_reward, label='Max Reward', color='green', linestyle='--', alpha=0.7)
    window_size = 20
    if len(history_avg_reward) >= window_size:
        moving_avg = np.convolve(history_avg_reward, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(history_avg_reward)), moving_avg,
                 label=f'Moving Avg ({window_size})', color='red', linewidth=2)
    ax1.axhline(y=best_reward, color='darkgreen', linestyle=':', label=f'Best Ever ({best_reward:.0f})')
    ax1.set_title(f'ARS-V2 Training: {run_name}')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Lap Completions per iteration
    ax2 = axes[0, 1]
    ax2.bar(range(len(history_lap_completions)), history_lap_completions,
            alpha=0.6, color='tab:green', label='Laps per Iteration')
    if len(history_lap_completions) >= 10:
        laps_moving_avg = np.convolve(history_lap_completions, np.ones(10)/10, mode='valid')
        ax2.plot(range(9, len(history_lap_completions)), laps_moving_avg,
                 label='Moving Avg (10)', linewidth=2, color='tab:red')
    ax2.set_title(f'Lap Completions per Iteration (Total: {total_laps})')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Laps Completed')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Cumulative Laps
    ax3 = axes[1, 0]
    cumulative_laps = np.cumsum(history_lap_completions)
    ax3.plot(cumulative_laps, label='Cumulative Laps', color='green', linewidth=2)
    ax3.fill_between(range(len(cumulative_laps)), cumulative_laps, alpha=0.3, color='green')
    ax3.set_title('Cumulative Lap Completions')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Total Laps')
    ax3.legend()
    ax3.grid(True)

    # Plot 4: Learning Rate and Noise decay
    ax4 = axes[1, 1]
    # Reconstruct LR and noise over iterations
    lrs = [learning_rate * (1.0 - 0.7 * (i / num_iterations)) for i in range(num_iterations)]
    noises = [noise * (1.0 - 0.7 * (i / num_iterations)) for i in range(num_iterations)]
    ax4.plot(lrs, label='Learning Rate', color='blue', linewidth=2)
    ax4.plot(noises, label='Noise', color='orange', linewidth=2)
    ax4.set_title('Hyperparameter Decay')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'{run_name}_plot.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Grafic salvat in: {plot_path}")

    env.close()


if __name__ == "__main__":
    train_ars_script()

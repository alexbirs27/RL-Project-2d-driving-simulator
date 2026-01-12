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
from src.agents.a2c.agent import A2CAgent

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
    f.write("episode,reward,loss,lap_completed,lap_time\n")

# Calea pentru CEL MAI BUN model
best_model_path = os.path.join(models_dir, f'{run_name}_BEST_model.pt')
# ---------------------

def train_a2c_script():
    # Initializare mediu
    env = make_env(render_mode=None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Hyperparameters for better learning
    num_episodes = 1000  # More episodes for better convergence
    lr = 3e-4  # Lower LR for stability

    print(f"Initializare A2C (state_dim={state_dim}, action_dim={action_dim}, LR={lr})...")
    agent = A2CAgent(state_dim=state_dim, action_dim=action_dim, lr=lr)

    history_rewards = []
    history_loss = []
    lap_completed_history = []
    lap_time_history = []
    offroad_steps = []

    # --- VARIABILA PENTRU SALVAREA CELUI MAI BUN MODEL ---
    best_reward = -float('inf')
    best_mean_reward = -float('inf')

    print(f"--- Incepe Antrenamentul A2C ({num_episodes} episoade) ---")

    for episode in range(num_episodes):
        # Update learning rate and entropy
        agent.update_learning_rate(episode, num_episodes)
        agent.update_entropy_coef(episode, num_episodes)

        state, _ = env.reset()
        done = False

        rewards = []
        log_probs = []
        states = []
        dones = []

        total_reward = 0
        lap_completed = False
        lap_time = None

        while not done:
            action, log_prob = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track off-road steps
            if 'on_road' in info:
                offroad_steps.append(0 if info.get('on_road', True) else 1)

            states.append(state)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)

            total_reward += reward
            state = next_state

            # Check for lap completion
            if terminated and info.get('lap_complete', False):
                lap_completed = True
                lap_time = info.get('lap_time', None)

        loss = agent.update(rewards, log_probs, states, dones, next_state)

        history_rewards.append(total_reward)
        history_loss.append(loss)
        lap_completed_history.append(1 if lap_completed else 0)
        if lap_time is not None:
            lap_time_history.append(lap_time)

        # Calculate statistics
        mean_reward = np.mean(history_rewards[-100:]) if len(history_rewards) >= 100 else np.mean(history_rewards)
        lap_completion_rate = np.mean(lap_completed_history[-100:]) if lap_completed_history else 0.0
        avg_lap_time = np.mean(lap_time_history[-50:]) if lap_time_history else None
        offroad_rate = np.mean(offroad_steps[-2000:]) if offroad_steps else 0.0

        # --- SALVARE CHECKPOINT INTELIGENT ---
        # Save based on mean reward (more stable)
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            agent.save(best_model_path)
            print(f"!!! RECORD NOU (mean): {best_mean_reward:.2f} -> Model salvat.")

        if total_reward > best_reward:
            best_reward = total_reward
        # -------------------------------------

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            lap_time_str = f"{avg_lap_time:.2f}s" if avg_lap_time is not None else "N/A"
            print(
                f"Ep {episode+1}/{num_episodes} | "
                f"Reward: {total_reward:.1f} | "
                f"Mean(100): {mean_reward:.1f} | "
                f"Best: {best_reward:.1f} | "
                f"Lap%: {lap_completion_rate:.2%} | "
                f"LapTime: {lap_time_str} | "
                f"Offroad: {offroad_rate:.2%}"
            )

        # Log to CSV
        with open(log_file, "a") as f:
            lt = lap_time if lap_time is not None else ""
            f.write(f"{episode+1},{total_reward},{loss},{1 if lap_completed else 0},{lt}\n")

    # Salvare model final
    final_path = os.path.join(models_dir, f'{run_name}_FINAL_model.pt')
    agent.save(final_path)
    print(f"\nAntrenament terminat!")
    print(f"Best Single Reward: {best_reward:.2f}")
    print(f"Best Mean Reward (100 ep): {best_mean_reward:.2f}")
    print(f"Total Laps Completed: {sum(lap_completed_history)}")

    # --- GENERARE GRAFICE ---
    print("Generare grafice...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Rewards
    ax1 = axes[0, 0]
    ax1.plot(history_rewards, label='Episode Reward', alpha=0.3, color='blue')
    window_size = 50
    if len(history_rewards) >= window_size:
        moving_avg = np.convolve(history_rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(len(moving_avg)), moving_avg, label=f'Moving Avg ({window_size})', color='red', linewidth=2)
    ax1.axhline(y=best_reward, color='green', linestyle='--', label=f'Best ({best_reward:.0f})')
    ax1.set_title(f'A2C Rewards: {run_name}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Loss
    ax2 = axes[0, 1]
    ax2.plot(history_loss, label='Training Loss', color='orange', alpha=0.5)
    if len(history_loss) >= window_size:
        loss_avg = np.convolve(history_loss, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(len(loss_avg)), loss_avg, label=f'Moving Avg ({window_size})', color='red', linewidth=2)
    ax2.set_title(f'A2C Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Lap Completions
    ax3 = axes[1, 0]
    # Calculate cumulative lap completions
    cumulative_laps = np.cumsum(lap_completed_history)
    ax3.plot(cumulative_laps, label='Cumulative Laps', color='green', linewidth=2)
    ax3.set_title(f'Lap Completions (Total: {sum(lap_completed_history)})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Laps Completed')
    ax3.legend()
    ax3.grid(True)

    # Plot 4: Lap Completion Rate over time
    ax4 = axes[1, 1]
    if len(lap_completed_history) >= 100:
        completion_rate = [np.mean(lap_completed_history[max(0, i-100):i+1])
                          for i in range(len(lap_completed_history))]
        ax4.plot(completion_rate, label='Lap Completion Rate (100 ep window)', color='purple', linewidth=2)
        ax4.set_ylim(0, 1)
    ax4.set_title('Lap Completion Rate')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Rate')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'{run_name}_plot.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Grafic salvat in: {plot_path}")

    env.close()

if __name__ == "__main__":
    train_a2c_script()

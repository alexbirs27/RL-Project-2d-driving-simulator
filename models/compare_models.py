import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch

# --- SETUP CĂI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Importuri agenți și mediu
from src.env.driving_env import make_env
from src.agents.random.random_agent import RandomAgent
from src.agents.ppo.agent import PPOAgent
from src.agents.ars.agent import ARSAgent
from src.agents.a2c.agent import A2CAgent

# Configurații
MODELS_DIR = os.path.join(project_root, 'models')
PLOTS_DIR = os.path.join(project_root, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_latest_model(prefix, extension):
    """Găsește cel mai recent fișier care începe cu 'prefix' în folderul models."""
    search_path = os.path.join(MODELS_DIR, f"{prefix}*{extension}")
    files = glob.glob(search_path)
    if not files:
        return None
    # Sortează după data modificării (cel mai nou primul)
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def evaluate_agent(env, agent, agent_name, episodes=10):
    """Rulează agentul pentru N episoade și returnează recompensele."""
    print(f"Evaluare {agent_name}...", end=" ", flush=True)
    rewards = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Logică unificată pentru a extrage acțiunea
            if agent_name == "Random":
                action = agent.act(state)
            elif agent_name == "PPO":
                action, _, _ = agent.act(state)
            elif agent_name == "A2C":
                action, _ = agent.act(state)
            elif agent_name == "ARS":
                action = agent.select_action(state)
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
        rewards.append(total_reward)
        print(".", end="", flush=True)
    
    print(f" Done! Medie: {np.mean(rewards):.2f}")
    return rewards

def main():
    print("--- COMPARARE MODELE ---")
    
    # 1. Inițializare Mediu
    # Folosim render_mode=None pentru viteză, sau 'human' dacă vrei să vezi cursele
    env = make_env(render_mode=None) 
    state_dim = 13
    action_dim = 5
    
    results = {}
    
    # --- 2. ÎNCĂRCARE AGENȚI ---
    
    # A. RANDOM
    random_agent = RandomAgent(env.action_space)
    results['Random'] = evaluate_agent(env, random_agent, "Random")

    # B. ARS
    ars_file = get_latest_model("ars", ".npy")
    if ars_file:
        print(f"Încărcare ARS: {os.path.basename(ars_file)}")
        ars_agent = ARSAgent(state_dim, action_dim)
        ars_agent.weights = np.load(ars_file)
        results['ARS'] = evaluate_agent(env, ars_agent, "ARS")
    else:
        print("Nu s-a găsit model ARS.")

    # C. PPO
    ppo_file = get_latest_model("ppo", ".pt")
    if ppo_file:
        print(f"Încărcare PPO: {os.path.basename(ppo_file)}")
        ppo_agent = PPOAgent(state_dim, action_dim)
        ppo_agent.load(ppo_file)
        results['PPO'] = evaluate_agent(env, ppo_agent, "PPO")
    else:
        print("Nu s-a găsit model PPO.")

    # D. A2C
    a2c_file = get_latest_model("a2c", ".pt")
    if a2c_file:
        print(f"Încărcare A2C: {os.path.basename(a2c_file)}")
        a2c_agent = A2CAgent(state_dim, action_dim)
        a2c_agent.load(a2c_file)
        results['A2C'] = evaluate_agent(env, a2c_agent, "A2C")
    else:
        print("Nu s-a găsit model A2C.")

    env.close()

    # --- 3. GENERARE GRAFIC COMPARATIV ---
    if not results:
        print("Nu am rezultate de comparat.")
        return

    print("\nGenerare grafic...")
    
    agents = list(results.keys())
    means = [np.mean(results[a]) for a in agents]
    stds = [np.std(results[a]) for a in agents] # Deviația standard (stabilitatea)

    plt.figure(figsize=(10, 6))
    
    # Creăm bar chart cu bare de eroare (yerr)
    bars = plt.bar(agents, means, yerr=stds, capsize=10, color=['gray', 'green', 'blue', 'orange'], alpha=0.7)
    
    plt.title('Comparare Performanță Agenți (Medie pe 10 Episoade)')
    plt.ylabel('Reward Total')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Adăugăm valorile pe bare
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom')

    # Salvare
    save_path = os.path.join(PLOTS_DIR, 'model_comparison.png')
    plt.savefig(save_path)
    print(f"Grafic salvat în: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
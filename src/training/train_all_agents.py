import sys
import os
import time
import glob # Necesar pentru a cauta fisierele model

# Setup cai import
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Config si module
from src.config import Config
from src.env.driving_env import make_env
from src.agents.random.random_agent import RandomAgent
from src.agents.ppo.agent import PPOAgent
from src.agents.ars.agent import ARSAgent
from src.agents.a2c.agent import A2CAgent

# Import scripturi de antrenament
try:
    from src.training.train_ppo import train_ppo_script
    from src.training.train_ars import train_ars_script
    from src.training.train_a2c import train_a2c_script
except ImportError as e:
    print(f"ATENTIE: Nu s-au putut importa scripturile de antrenament: {e}")

# Cai catre foldere
MODELS_DIR = os.path.join(project_root, 'models')

def get_latest_model(prefix, extension):
    """
    Gaseste cel mai recent fisier din folderul models care incepe cu prefixul dat.
    Ex: prefix='ars' -> gaseste 'ars_run_2023...weights.npy'
    """
    search_path = os.path.join(MODELS_DIR, f"{prefix}*{extension}")
    files = glob.glob(search_path)
    if not files:
        return None
    # Sorteaza dupa data modificarii (cel mai nou primul)
    return max(files, key=os.path.getmtime)

def train_all():
    print("\n--- MENIU PRINCIPAL ---")
    print("Selecteaza optiunea:")
    print("1. TESTARE: Random Agent")
    print("2. TESTARE: PPO Agent (Evaluare)")
    print("3. TESTARE: ARS Agent (Evaluare)")
    print("4. TESTARE: A2C Agent (Evaluare)")
    print("-" * 30)
    print("5. ANTRENAMENT COMPLET (Secvential: PPO -> ARS -> A2C)")
    
    choice = input("\nIntrodu numarul (1-5): ")
    
    # --- LOGICA PENTRU TESTARE (RENDER ON) ---
    if choice in ['1', '2', '3', '4']:
        # Cream mediul DOAR pentru testare vizuala
        env = make_env(render_mode="human")
        
        if choice == '1':
            print("Rulare Agent Random...")
            agent = RandomAgent(env.action_space)
            run_evaluation(env, agent, "random")
            
        elif choice == '2':
            print("Initializare PPO pentru testare...")
            agent = PPOAgent(state_dim=Config.STATE_DIM, action_dim=Config.ACTION_DIM)
            
            # Cautam cel mai recent model PPO
            model_path = get_latest_model("ppo", ".pt")
            
            if model_path:
                agent.load(model_path)
                print(f"Model PPO incarcat din: {os.path.basename(model_path)}")
            else:
                print("ATENTIE: Nu s-a gasit model PPO! Se ruleaza cu greutati random.")
            run_evaluation(env, agent, "ppo")

        elif choice == '3':
            print("Initializare ARS pentru testare...")
            agent = ARSAgent(state_dim=Config.STATE_DIM, action_dim=Config.ACTION_DIM)
            
            # Cautam cel mai recent model ARS (.npy)
            model_path = get_latest_model("ars", ".npy")
            
            if model_path:
                import numpy as np
                agent.weights = np.load(model_path)
                print(f"Model ARS incarcat din: {os.path.basename(model_path)}")
            else:
                 print("ATENTIE: Nu s-a gasit model ARS!")
            run_evaluation(env, agent, "ars")
            
        elif choice == '4':
            print("Initializare A2C pentru testare...")
            agent = A2CAgent(state_dim=Config.STATE_DIM, action_dim=Config.ACTION_DIM)
            
            # Cautam mai intai modelul BEST, daca nu, pe cel normal
            model_path = get_latest_model("a2c", "BEST_model.pt")
            if not model_path:
                model_path = get_latest_model("a2c", ".pt")

            if model_path:
                agent.load(model_path)
                print(f"Model A2C incarcat din: {os.path.basename(model_path)}")
            else:
                print("ATENTIE: Nu s-a gasit model A2C!")
            run_evaluation(env, agent, "a2c")
            
        env.close()

    # --- LOGICA PENTRU ANTRENAMENT (RENDER OFF) ---
    elif choice == '5':
        print("\n" + "="*50)
        print("  INCEPERE CICLU COMPLET DE ANTRENAMENT")
        print("="*50 + "\n")

        # 1. PPO
        print("\n>>> ETAPA 1/3: Antrenament PPO...")
        try:
            train_ppo_script() 
            print(">>> PPO Finalizat cu succes.")
        except Exception as e:
            print(f"!!! Eroare la antrenarea PPO: {e}")

        time.sleep(2) # Pauza scurta pt scriere fisiere

        # 2. ARS
        print("\n>>> ETAPA 2/3: Antrenament ARS...")
        try:
            train_ars_script()
            print(">>> ARS Finalizat cu succes.")
        except Exception as e:
            print(f"!!! Eroare la antrenarea ARS: {e}")

        time.sleep(2)

        # 3. A2C
        print("\n>>> ETAPA 3/3: Antrenament A2C...")
        try:
            train_a2c_script()
            print(">>> A2C Finalizat cu succes.")
        except Exception as e:
            print(f"!!! Eroare la antrenarea A2C: {e}")

        print("\n" + "="*50)
        print("  CICLU DE ANTRENAMENT FINALIZAT")
        print("  Verifica folderele /plots si /logs pentru rezultate.")
        print("="*50)

    else:
        print("Optiune invalida!")

def run_evaluation(env, agent, agent_type):
    """Ruleaza cateva episoade de test."""
    # Numarul de episoade de test poate fi luat si din Config
    episodes = getattr(Config, 'EVAL_EPISODES', 3)
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        print(f"Start Episod {ep+1}...")
        
        while not done:
            if agent_type == "random":
                action = agent.act(obs)
            elif agent_type == "ppo":
                action, _, _ = agent.act(obs)
            elif agent_type == "ars":
                # ARS nu are nevoie de noise la testare
                action = agent.select_action(obs)
            elif agent_type == "a2c":
                action, _ = agent.act(obs)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()
            
        print(f"Episod {ep+1}: Reward {total_reward:.2f}")

if __name__ == "__main__":
    train_all()
import torch
import numpy as np
from environment import RacingEnv
from agent import A2CAgent
import os

def visualize_best_model(model_path='models/best_model.pt'):
    # 1. Inițializăm mediul în mod 'human' pentru a vedea mașina
    env = RacingEnv(render_mode='human', max_steps=1000)
    
    # 2. Inițializăm agentul
    # Notă: obs_dim și action_dim trebuie să coincidă cu cele de la antrenare
    agent = A2CAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    # 3. Încărcăm greutățile antrenate
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Succes! Modelul {model_path} a fost încărcat.")
    else:
        print(f"Eroare: Nu am găsit fișierul la {model_path}")
        return

    # 4. Rulăm o simulare
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    print("Rulăm simularea... Apasă Ctrl+C în consolă pentru a opri.")
    
    try:
        while not done:
            # Selectăm acțiunea determinist (fără explorare) pentru performanță maximă
            action, _, _ = agent.select_action(state, deterministic=True)
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            env.render()
            
        print(f"Simulare terminată! Recompensă totală: {total_reward:.2f}")
    except KeyboardInterrupt:
        print("\nVizualizare oprită de utilizator.")
    finally:
        env.close()

if __name__ == "__main__":
    # Verifică dacă ai modelul în 'models' sau 'demo_models'
    path = 'models/best_model.pt'
    if not os.path.exists(path):
        path = 'demo_models/best_model.pt'
        
    visualize_best_model(path)
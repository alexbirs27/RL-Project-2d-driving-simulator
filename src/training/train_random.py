import time
import sys
import os
# Adaugă folderul rădăcină al proiectului în sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import make_env

def run_random_agent(episodes=5, render=True):
    # Creăm mediul folosind funcția ta ajutătoare
    # Folosim render_mode="human" pentru a vedea ce face mașina
    render_mode = "human" if render else None
    env = make_env(render_mode=render_mode)

    print(f"Începem testarea pentru {episodes} episoade...")

    for ep in range(episodes):
        observation, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0

        while not (terminated or truncated):
            # 1. Agentul alege o acțiune complet aleatorie
            action = env.action_space.sample()

            # 2. Trimitem acțiunea în environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1

            # 3. Randare vizuală
            if render:
                env.render()
                # Opțional: adăugăm o mică pauză dacă simularea rulează prea repede
                # time.sleep(0.01)

            if steps % 100 == 0:
                print(f"Episod {ep+1} | Pas {steps} | Reward curent: {total_reward:.2f}")

        print(f"--- Episodul {ep+1} s-a terminat ---")
        print(f"Pași totali: {steps}")
        print(f"Reward total: {total_reward:.2f}")
        print(f"Lap Complete: {info['lap_complete']}")
        print("-" * 30)

    env.close()

if __name__ == "__main__":
    run_random_agent(episodes=3, render=True)
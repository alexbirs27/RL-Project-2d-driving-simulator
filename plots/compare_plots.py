import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Setup căi
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

LOGS_DIR = os.path.join(project_root, 'logs')
PLOTS_DIR = os.path.join(project_root, 'plots')

def load_data():
    all_files = glob.glob(os.path.join(LOGS_DIR, "*.csv"))
    data = {}
    
    print(f"Caut în {LOGS_DIR}...")
    
    for filename in all_files:
        name = os.path.basename(filename)
        
        # Încercăm să citim CSV-ul
        try:
            df = pd.read_csv(filename)
            
            # Identificăm tipul agentului din nume
            if "ppo" in name.lower():
                label = f"PPO ({name})"
                # PPO/A2C au 'episode' și 'reward' (sau avg_reward în unele versiuni)
                # Standardizăm coloanele pentru grafic
                x = df.iloc[:, 0] # Prima coloană e mereu timpul (epoch/episode/iteration)
                y = df.iloc[:, 1] # A doua coloană e mereu reward-ul principal
            elif "a2c" in name.lower():
                label = f"A2C ({name})"
                x = df.iloc[:, 0]
                y = df.iloc[:, 1]
            elif "ars" in name.lower():
                label = f"ARS ({name})"
                x = df.iloc[:, 0]
                y = df['avg_reward'] # ARS are avg_reward si max_reward
            else:
                continue

            data[label] = (x, y)
            print(f" -> Încărcat: {name}")
            
        except Exception as e:
            print(f" ! Eroare la citirea {name}: {e}")
            
    return data

def plot_comparison():
    data = load_data()
    
    if not data:
        print("Nu am găsit fișiere log valide (.csv) in folderul logs/.")
        return

    plt.figure(figsize=(12, 8))
    
    for label, (x, y) in data.items():
        # Aplicăm o netezire (Moving Average) dacă sunt multe puncte, ca să se vadă clar trendul
        window = 20 if len(y) > 100 else 1
        y_smooth = y.rolling(window=window, min_periods=1).mean()
        
        plt.plot(x, y_smooth, label=label, linewidth=2)

    plt.title("Comparație Istoric Antrenament (Learning Curves)", fontsize=16)
    plt.xlabel("Episoade / Iterații", fontsize=12)
    plt.ylabel("Reward Mediu", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Salvare
    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, 'history_comparison.png')
    plt.savefig(save_path)
    print(f"\nGrafic salvat în: {save_path}")
    plt.show()

if __name__ == "__main__":
    # Verificăm dacă avem pandas instalat, e necesar pentru CSV rapid
    try:
        import pandas
    except ImportError:
        print("Acest script necesită pandas. Rulează: pip install pandas")
        sys.exit(1)
        
    plot_comparison()
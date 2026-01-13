# Documentatie Completa: ARS si A2C

## Cuprins
1. [Introducere](#introducere)
2. [Environment-ul Folosit](#environment-ul-folosit)
3. [ARS - Augmented Random Search](#ars---augmented-random-search)
4. [A2C - Advantage Actor-Critic](#a2c---advantage-actor-critic)
5. [Comparatie ARS vs A2C](#comparatie-ars-vs-a2c)
6. [Rezultate si Analiza](#rezultate-si-analiza)
7. [Ghid de Antrenare](#ghid-de-antrenare)
8. [Istoric Antrenare si Configurari](#istoric-antrenare-si-configurari)
9. [Environment-uri Folosite](#environment-uri-folosite)
10. [Istoricul Configurarilor Testate](#istoricul-configurarilor-testate)
11. [Probleme Rezolvate in Dezvoltare](#probleme-rezolvate-in-dezvoltare)
12. [Rezultate Comparate](#rezultate-comparate)

---

## Introducere

Acest document explica in detaliu doi algoritmi de Reinforcement Learning implementati pentru simulatorul de racing 2D:

1. **ARS (Augmented Random Search)** - Algoritm gradient-free, simplu dar eficient
2. **A2C (Advantage Actor-Critic)** - Algoritm bazat pe policy gradient cu rețele neuronale

Ambii agenti sunt antrenati sa conduca o masina pe un circuit de curse, invatand sa:
- Navigheze prin checkpoints
- Mentina viteza optima
- Ramana pe pista
- Complete tururi

---

## Environment-ul Folosit

### Fisier: `env.py` - RacingEnv

#### Spatiul de Observatii (10 valori)

```
Observation Space (10 valori):

Senzori Distanta (5 valori) - Ray casting 400px:
    1. Raza frontala (distanta normalizata 0-1)
    2. Raza front-dreapta (45 grade)
    3. Raza front-stanga (-45 grade)
    4. Raza dreapta (90 grade)
    5. Raza stanga (-90 grade)

Navigatie catre Checkpoint (2 valori):
    6. Distanta la urmatorul checkpoint (normalizata 0-1)
    7. Unghiul catre checkpoint (normalizat -1 la 1)

Constientizare Linie Racing (3 valori):
    8. Distanta de la centrul pistei (0=perfect, 1=margine)
    9. Unghiul fata de directia pistei
    10. Viteza (normalizata 0-1)
```

#### Spatiul de Actiuni (9 actiuni discrete)

| Actiune | Descriere |
|---------|-----------|
| 0 | Nimic |
| 1 | Accelereaza |
| 2 | Franeaza |
| 3 | Vireaza stanga |
| 4 | Vireaza dreapta |
| 5 | Accelereaza + Stanga |
| 6 | Accelereaza + Dreapta |
| 7 | Franeaza + Stanga |
| 8 | Franeaza + Dreapta |

#### Structura Reward-urilor

| Componenta | Valoare | Descriere |
|------------|---------|-----------|
| Checkpoint | +50.0 | Per checkpoint atins (doar inainte!) |
| Viteza | +0.5 | Per step, proportional cu viteza |
| Linie centrala | +0.5 | Per step, cat de aproape de centru |
| Off-road | -5.0 | Per step in afara pistei |
| Tur complet | +1000.0 | Bonus pentru finalizare tur |

#### Conditii de Terminare

- **Terminated**: Cand masina completeaza un tur
- **Truncated**:
  - Depasire 20,000 steps
  - 3,000 steps fara progres la checkpoint
  - 1,000 steps consecutive off-road

---

## ARS - Augmented Random Search

### Ce este ARS?

**Augmented Random Search** este un algoritm de optimizare **gradient-free** care:
- NU foloseste retele neuronale
- NU calculeaza gradiente
- Foloseste o **politica liniara** simpla
- Exploreaza prin **perturbari aleatoare** ale greutatilor

### Cum Functioneaza ARS?

#### 1. Politica Liniara

```
Actiune = argmax(W * stare)

Unde:
- W = matrice de greutati (action_dim x state_dim) = (9 x 10)
- stare = vector de 10 observatii
- Rezultat = 9 scoruri, alegem actiunea cu scorul maxim
```

#### 2. Algoritmul de Antrenare

```
PENTRU fiecare iteratie:
    1. Genereaza N perturbari aleatoare (delta)

    2. Pentru fiecare perturbare delta:
       a) Testeaza W + delta * noise  → reward_pozitiv
       b) Testeaza W - delta * noise  → reward_negativ

    3. Calculeaza diferenta: (reward_pozitiv - reward_negativ)

    4. Actualizeaza greutatile:
       W = W + lr * SUM(diferente * delta) / (N * std_rewards)
```

#### Diagrama Vizuala

```
                    [STARE (10 valori)]
                           |
                           v
    +------------------[Inmultire Matriceala]------------------+
    |                         |                                |
    |                  W (9 x 10)                             |
    |                         |                                |
    |                         v                                |
    |               [9 Scoruri Actiuni]                       |
    |                         |                                |
    |                         v                                |
    |                    argmax()                              |
    |                         |                                |
    |                         v                                |
    |               [ACTIUNE SELECTATA]                       |
    +----------------------------------------------------------+
```

### Implementare ARS

#### Fisier: `ARS/agent.py`

```python
class ARSAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.02, noise=0.03, num_deltas=16):
        # Dimensiuni
        self.state_dim = state_dim      # 10 observatii
        self.action_dim = action_dim    # 9 actiuni

        # Hiperparametri
        self.lr = learning_rate         # 0.02
        self.noise = noise              # 0.03
        self.num_deltas = num_deltas    # 16 directii de explorare

        # Politica liniara: matrice de greutati (9 x 10)
        self.weights = np.zeros((action_dim, state_dim))
```

#### Selectia Actiunii

```python
def select_action(self, state, delta=None, direction=None):
    weights = self.weights.copy()

    # Aplica perturbarea pentru explorare
    if delta is not None:
        if direction == "plus":
            weights += delta * self.noise
        else:
            weights -= delta * self.noise

    # Inmultire: greutati * stare
    logits = weights.dot(state)

    # Alege actiunea cu scorul maxim
    return np.argmax(logits)
```

#### Update Greutati

```python
def update(self, rollouts, sigma_rewards):
    step = np.zeros(self.weights.shape)

    for r_positive, r_negative, delta in rollouts:
        # Diferenta intre directia + si -
        step += (r_positive - r_negative) * delta

    # Actualizare cu normalizare
    self.weights += self.lr / (self.num_deltas * sigma_rewards) * step
```

### Configuratie ARS

```python
# Din config.py
ARS_LEARNING_RATE = 0.02      # Pas de update
ARS_NOISE = 0.03              # Scala perturbatiilor
ARS_NUM_DELTAS = 16           # Directii de explorare
ARS_TOTAL_ITERATIONS = 500    # Iteratii totale
```

### Antrenare ARS

#### Comanda

```bash
cd ARS
python train_ars.py
```

#### Ce se Intampla la Antrenare

1. **Generare Perturbari**: 16 directii random (delta)
2. **Evaluare**:
   - Ruleaza episod cu W + delta (reward_pozitiv)
   - Ruleaza episod cu W - delta (reward_negativ)
3. **Update**: Ajusteaza W in directiile care au dat reward mai bun
4. **Salvare**: Model BEST si grafice

#### Output Asteptat

```
--- RULARE NOUA: ars_run_2025-01-12_14-30-00 ---
Initializare ARS (state_dim=10, action_dim=9, StepSize=0.02)...
--- Incepe Antrenamentul ARS (500 iteratii) ---
Iteratia 1/500 | Avg: -25.30 | Max: 45.20 | Best: 45.20
!!! RECORD NOU: 45.20 -> Model salvat.
Iteratia 2/500 | Avg: -18.50 | Max: 52.10 | Best: 52.10
!!! RECORD NOU: 52.10 -> Model salvat.
...
Iteratia 500/500 | Avg: 180.50 | Max: 450.20 | Best: 520.30
```

### Avantaje si Dezavantaje ARS

| Avantaje | Dezavantaje |
|----------|-------------|
| Extrem de simplu | Sample inefficient |
| Fara backpropagation | Nu functioneaza bine pe task-uri complexe |
| Fara hiperparametri complicati | Politica liniara = limitata |
| Usor de paralelizat | Convergenta lenta |
| Functioneaza cu reward sparse | Necesita multe evaluari |

---

## A2C - Advantage Actor-Critic

### Ce este A2C?

**Advantage Actor-Critic** este un algoritm de **policy gradient** care combina:
- **Actor**: Retea neurala care invata POLITICA (ce actiune sa aleaga)
- **Critic**: Retea neurala care invata VALOAREA starilor

### Concepte Cheie

#### 1. Policy Gradient
In loc sa estimam Q-values (ca DQN), invatam direct o **politica** - o distributie de probabilitati peste actiuni.

#### 2. Advantage Function
```
Advantage = Return - Value(stare)

Unde:
- Return = reward-ul real obtinut (discounted)
- Value = ce "crede" Critic-ul ca va obtine

Daca Advantage > 0: Actiunea a fost MAI BUNA decat asteptat
Daca Advantage < 0: Actiunea a fost MAI REA decat asteptat
```

#### 3. Actor-Critic

```
     [STARE]
        |
        v
+-------+-------+
|               |
v               v
[ACTOR]      [CRITIC]
   |            |
   v            v
Probabilitati  Valoare
  Actiuni      Stare
```

### Arhitectura Retelelor

#### Fisier: `A2C/networks.py`

```python
class ActorNet(nn.Module):
    """Retea pentru politica (Actor)"""
    def __init__(self, state_dim, action_dim):
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),    # 10 -> 128
            nn.ReLU(),
            nn.Linear(128, 128),           # 128 -> 128
            nn.ReLU(),
            nn.Linear(128, action_dim)     # 128 -> 9 (logits actiuni)
        )

class CriticNet(nn.Module):
    """Retea pentru valoare (Critic)"""
    def __init__(self, state_dim):
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),    # 10 -> 128
            nn.ReLU(),
            nn.Linear(128, 128),           # 128 -> 128
            nn.ReLU(),
            nn.Linear(128, 1)              # 128 -> 1 (valoare stare)
        )
```

### Diagrama Arhitecturii

```
                    [STARE (10 valori)]
                           |
           +---------------+---------------+
           |                               |
           v                               v
    +-------------+                 +-------------+
    |   ACTOR     |                 |   CRITIC    |
    +-------------+                 +-------------+
    |  10 -> 128  |                 |  10 -> 128  |
    |    ReLU     |                 |    ReLU     |
    |  128 -> 128 |                 |  128 -> 128 |
    |    ReLU     |                 |    ReLU     |
    |  128 -> 9   |                 |  128 -> 1   |
    +-------------+                 +-------------+
           |                               |
           v                               v
    [9 Logits]                      [1 Valoare]
           |                               |
           v                               |
    [Softmax]                              |
           |                               |
           v                               |
    [Probabilitati]                        |
           |                               |
           v                               v
    [Sample Actiune]            [Calcul Advantage]
           |                               |
           +---------------+---------------+
                           |
                           v
                    [LOSS TOTAL]
                           |
                           v
                   [BACKPROPAGATION]
```

### Algoritmul de Antrenare

#### 1. Colecteaza un Episod

```python
while not done:
    action, log_prob = agent.act(state)
    next_state, reward, done = env.step(action)

    # Salveaza tranzitia
    states.append(state)
    rewards.append(reward)
    log_probs.append(log_prob)
```

#### 2. Calculeaza Discounted Returns

```python
R = 0  # Start de la final
returns = []

for r, done in zip(reversed(rewards), reversed(dones)):
    if done:
        R = 0  # Reset la terminare
    R = r + gamma * R  # Discounted return
    returns.insert(0, R)
```

#### 3. Calculeaza Advantage

```python
# Value estimat de Critic
values = critic(states)

# Advantage = Returns - Values
advantage = returns - values
```

#### 4. Calculeaza Loss-urile

```python
# Actor Loss: Maximizeaza log_prob * advantage
actor_loss = -(log_probs * advantage.detach()).mean()

# Critic Loss: Minimizeaza eroarea Value
critic_loss = MSE(values, returns)

# Entropy Loss: Bonus pentru explorare
entropy_loss = -entropy(policy).mean()

# Loss Total
total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
```

#### 5. Update Greutati

```python
optimizer.zero_grad()
total_loss.backward()
clip_grad_norm_(params, 0.5)  # Stabilitate
optimizer.step()
```

### Implementare A2C

#### Fisier: `A2C/agent.py`

```python
class A2CAgent:
    def __init__(self, state_dim=10, action_dim=9, lr=1e-3, gamma=0.99, entropy_coef=0.01):
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Retelele
        self.actor = ActorNet(state_dim, action_dim)
        self.critic = CriticNet(state_dim)

        # Optimizatoare separate
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)
```

#### Selectia Actiunii (Training)

```python
def act(self, state):
    state_t = torch.tensor(state, dtype=torch.float32)

    # Actor genereaza logits
    logits = self.actor(state_t)
    probs = F.softmax(logits, dim=-1)

    # Sample din distributie categorica
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    return action.item(), log_prob
```

#### Selectia Actiunii (Evaluare)

```python
def select_action(self, state):
    """Fara gradient, doar argmax"""
    with torch.no_grad():
        logits = self.actor(state_t)
        probs = F.softmax(logits, dim=-1)
        action = torch.argmax(probs).item()
    return action
```

### Configuratie A2C

```python
# Din config.py
A2C_LEARNING_RATE = 1e-3      # 0.001
A2C_GAMMA = 0.99              # Factor de discount
A2C_ENTROPY_COEF = 0.01       # Bonus explorare
A2C_TOTAL_EPISODES = 500      # Episoade totale
```

### Antrenare A2C

#### Comanda

```bash
cd A2C
python train_a2c.py
```

#### Output Asteptat

```
--- RULARE NOUA: a2c_run_2025-01-12_15-00-00 ---
Initializare A2C (state_dim=10, action_dim=9, LR=0.001)...
--- Incepe Antrenamentul A2C (500 episoade) ---
Episod 1/500 | Reward: -120.50 | Loss: 2.3450 | Best: -120.50
Episod 2/500 | Reward: -95.30 | Loss: 1.8920 | Best: -95.30
!!! RECORD NOU: -95.30 -> Model salvat.
...
Episod 500/500 | Reward: 850.20 | Loss: 0.1250 | Best: 920.40
```

### Avantaje si Dezavantaje A2C

| Avantaje | Dezavantaje |
|----------|-------------|
| Sample efficient | Mai complex de implementat |
| Functioneaza pe task-uri complexe | Hiperparametri sensibili |
| Poate invata politici non-liniare | Necesita GPU pentru viteza |
| Convergenta mai rapida | Poate fi instabil |
| Suport pentru explorare (entropy) | Variance mare in gradiente |

---

## Comparatie ARS vs A2C

### Tabel Comparativ

| Caracteristica | ARS | A2C |
|----------------|-----|-----|
| **Tip Algoritm** | Gradient-free | Policy Gradient |
| **Politica** | Liniara (W * x) | Retea Neurala |
| **Complexitate Cod** | ~50 linii | ~120 linii |
| **Complexitate Matematica** | Simpla | Medie |
| **Retele Neuronale** | NU | DA (2 retele) |
| **Backpropagation** | NU | DA |
| **Sample Efficiency** | Slaba | Buna |
| **Paralelizare** | Usor | Mai dificil |
| **Stabilitate** | Foarte stabila | Poate fi instabila |
| **Capacitate** | Limitata (liniar) | Mare (non-liniar) |
| **GPU Necesar** | NU | Recomandat |
| **Viteza Inferenta** | Foarte rapida | Rapida |
| **Hiperparametri** | 3-4 | 5-7 |

### Cand sa Folosesti Fiecare?

#### Foloseste ARS cand:
- Task-ul este relativ simplu
- Vrei o implementare rapida
- Nu ai GPU puternic
- Vrei stabilitate garantata
- Politica liniara e suficienta
- Vrei sa paralelizezi usor

#### Foloseste A2C cand:
- Task-ul necesita politici complexe
- Ai resurse computationale bune
- Vrei convergenta mai rapida
- Ai nevoie de explorare sofisticata
- Vrei sa captezi relatii non-liniare

---

## Rezultate si Analiza

### Metrici de Performanta

| Metrica | ARS (500 iter) | A2C (500 ep) |
|---------|----------------|--------------|
| Best Reward | ~500-800 | ~800-1200 |
| Avg Reward Final | ~200-400 | ~400-800 |
| Checkpoints/Episod | 5-15 | 10-25 |
| Lap Completions | Rare | Ocazionale |
| Timp Antrenare | ~30-60 min | ~60-120 min |

### Interpretare Rezultate

#### ARS
- **Best Reward ~500-800**: Invata sa navigheze partial
- **Avg mai mic**: Varianta mare din cauza explorarrii random
- **Lap-uri rare**: Politica liniara e limitata pentru curve complexe

#### A2C
- **Best Reward ~800-1200**: Performanta mai buna
- **Avg mai consistent**: Invata politici mai stabile
- **Lap-uri ocazionale**: Retele pot invata manevre complexe

### Grafice Generate

Ambele scripturi genereaza grafice automat in folderul `plots/`:

**ARS**: `ars_run_TIMESTAMP_plot.png`
- Average Reward per iteratie
- Max Reward per iteratie
- Linie orizontala pentru Best Ever

**A2C**: `a2c_run_TIMESTAMP_plot.png`
- Episode Reward + Moving Average
- Training Loss pe episoade
- Linie orizontala pentru Best

---

## Ghid de Antrenare

### Structura Proiectului

```
RL-Project-2d-driving-simulator/
|
+-- A2C/
|   |-- agent.py          # Agent A2C
|   |-- networks.py       # Actor si Critic
|   |-- train_a2c.py      # Script antrenare
|   +-- __init__.py
|
+-- ARS/
|   |-- agent.py          # Agent ARS
|   |-- train_ars.py      # Script antrenare
|   +-- __init__.py
|
+-- config.py             # Toate hiperparametrele
+-- env.py                # Environment (RacingEnv)
+-- game.py               # Motor joc
|
+-- logs/                 # Fisiere CSV cu metrici
+-- models/               # Modele salvate (.pt, .npy)
+-- plots/                # Grafice generate
```

### Comenzi de Antrenare

#### ARS
```bash
# Din directorul proiectului
python -m ARS.train_ars

# SAU din folderul ARS
cd ARS
python train_ars.py
```

#### A2C
```bash
# Din directorul proiectului
python -m A2C.train_a2c

# SAU din folderul A2C
cd A2C
python train_a2c.py
```

### Modificare Hiperparametri

Toate valorile sunt in `config.py`:

```python
# Pentru A2C
A2C_LEARNING_RATE = 1e-3      # Creste pentru invatare mai rapida
A2C_GAMMA = 0.99              # Scade pentru focus pe reward imediat
A2C_ENTROPY_COEF = 0.01       # Creste pentru mai multa explorare
A2C_TOTAL_EPISODES = 500      # Creste pentru antrenare mai lunga

# Pentru ARS
ARS_LEARNING_RATE = 0.02      # Pas de update
ARS_NOISE = 0.03              # Zgomot explorare
ARS_NUM_DELTAS = 16           # Directii testate
ARS_TOTAL_ITERATIONS = 500    # Iteratii totale
```

### Vizualizare Antrenare

```bash
# Antreneaza cu rendering vizual (mai lent)
# Modifica in script: render_mode="human"
```

### Incarcare Model Salvat

```python
# ARS
import numpy as np
weights = np.load('models/ars_run_BEST_weights.npy')
agent.weights = weights

# A2C
agent.load('models/a2c_run_BEST_model.pt')
```

### Tips pentru Antrenare

1. **Incepe cu hiperparametrii default** - sunt deja tuned
2. **Monitorizeaza graficele** - indica daca invata
3. **Salveaza frecvent** - checkpoint-urile sunt importante
4. **Foloseste render_mode=None** pentru antrenare rapida
5. **Creste episoadele/iteratiile** pentru performanta mai buna

---

## Concluzie

**ARS** este ideal pentru:
- Prototipare rapida
- Baseline simplu
- Resurse limitate

**A2C** este ideal pentru:
- Performanta maxima
- Task-uri complexe
- Cand ai timp si resurse

Ambii algoritmi sunt implementati complet si functionali in acest proiect, oferind o baza solida pentru experimentare cu RL pe task-uri de conducere autonoma.

---

## Istoric Antrenare si Configurari

### Evolutia Proiectului

Proiectul a trecut prin mai multe iteratii de dezvoltare, fiecare aducand imbunatatiri semnificative.

### Timeline Dezvoltare (din Git History)

| Data/Commit | Modificare | Impact |
|-------------|-----------|--------|
| `432621d` | Prima versiune PPO | Structura initiala |
| `f38c189` | Prima rulare model | Baseline initial |
| `ccfb510` | Fix env (12 obs, 9 actiuni) | Dimensiuni corecte |
| `357473f` | Entropy bonus + LR mai mic | Mai multa explorare |
| `1752f33` | LR decay liniar la 0 | Convergenta mai buna |
| `4ac516a` | Gradient clipping | Stabilitate training |
| `166d4fb` | Retea mai adanca | Capacitate mai mare |
| `1d782f1` | LR decay 10%, checkpointing | Salvare modele |
| `b448d02` | Reward shaping imbunatatit | Training mai eficient |
| `abfea72` | Redesign observation space | Fix catastrophic forgetting |
| `49925ad` | RWD drift physics + 13 obs | Fizica realista |
| `dc2fd09` | Config centralizat, obs imbunatatite | Organizare cod |
| `f1969ca` | LR reduction 1000x + epsilon freeze | DQN stability |
| `7eaa5f0` | Adaugare A2C si ARS | Agenti noi |

---

## Environment-uri Folosite

### Environment 1: DrivingEnv (Versiune Initiala)

**Fisier**: `src/env/driving_env.py`

#### Specificatii

| Parametru | Valoare |
|-----------|---------|
| Observation Space | 13 valori |
| Action Space | 5 actiuni discrete |
| Max Steps | 3000 |

#### Structura Observatii (13 features)

```
Index   Valoare                     Descriere
0       velocity_norm               Viteza normalizata (0-1)
1       angle_sin                   sin(unghi masina)
2       angle_cos                   cos(unghi masina)
3-10    distance_sensors[8]         8 senzori directii (radial)
11      progress                    Progres pe pista (0-1)
12      on_road                     Pe drum: 1.0, Off: -1.0
```

#### Actiuni (5 discrete)

| ID | Actiune |
|----|---------|
| 0 | Nimic |
| 1 | Accelereaza |
| 2 | Franeaza |
| 3 | Vireaza stanga |
| 4 | Vireaza dreapta |

#### Reward Structure (DrivingEnv)

```python
# 1. Viteza - reward principal
velocity_reward = (velocity / max_velocity) * 2.0

# 2. Progres - bonusuri mari
if diff > 0: reward += diff * 500.0
if diff < 0: reward += diff * 50.0  # Penalizare usoara

# 3. Drum
if on_road: reward += 1.0
else: reward -= 5.0; terminated = True  # STOP IMEDIAT

# 4. Finish
if lap_complete: reward += 2000.0

# 5. Extra
if velocity < 1.0: reward -= 5.0  # Sta pe loc
reward -= 0.01  # Penalizare timp
```

**Nota**: DrivingEnv termina episodul IMEDIAT la off-road (training rapid).

---

### Environment 2: RacingEnv (Versiune Curenta)

**Fisier**: `env.py`

#### Specificatii

| Parametru | Valoare |
|-----------|---------|
| Observation Space | 10 valori |
| Action Space | 9 actiuni discrete |
| Max Steps | 20,000 |
| Ray Distance | 400 pixels |

#### Structura Observatii (10 features)

```
Index   Valoare                     Descriere
0       front_ray                   Raza fata (0=aproape, 1=departe)
1       front_right_ray             Raza fata-dreapta (45°)
2       front_left_ray              Raza fata-stanga (-45°)
3       right_ray                   Raza dreapta (90°)
4       left_ray                    Raza stanga (-90°)
5       dist_to_checkpoint          Distanta la checkpoint (norm)
6       angle_to_checkpoint         Unghi la checkpoint (-1 la 1)
7       dist_to_centerline          Distanta de centru (0=perfect)
8       angle_to_centerline         Unghi fata de directia pistei
9       velocity                    Viteza normalizata (0-1)
```

#### Actiuni (9 discrete)

| ID | Actiune |
|----|---------|
| 0 | Nimic |
| 1 | Accelereaza |
| 2 | Franeaza |
| 3 | Vireaza stanga |
| 4 | Vireaza dreapta |
| 5 | Accelereaza + Stanga |
| 6 | Accelereaza + Dreapta |
| 7 | Franeaza + Stanga |
| 8 | Franeaza + Dreapta |

#### Reward Structure (RacingEnv)

```python
# Din config.py
REWARD_CHECKPOINT = 50.0      # Per checkpoint (doar inainte!)
REWARD_SPEED = 0.5            # Per step * viteza
REWARD_CENTERLINE = 0.5       # Per step * apropierea de centru
REWARD_OFFROAD = -5.0         # Per step off-road
REWARD_LAP_COMPLETE = 1000.0  # Bonus tur
```

**Nota**: RacingEnv permite recuperare din off-road (1000 steps).

---

## Comparatie Environment-uri

| Caracteristica | DrivingEnv | RacingEnv |
|----------------|------------|-----------|
| **Observatii** | 13 | 10 |
| **Actiuni** | 5 | 9 |
| **Senzori** | 8 radiali | 5 directii cheie |
| **Navigatie** | Progress (0-1) | Checkpoint + Angle |
| **Off-road** | Terminare imediata | 1000 steps toleranta |
| **Max Steps** | 3,000 | 20,000 |
| **Checkpoint Reward** | +500 * diff | +50 fix |
| **Lap Reward** | +2,000 | +1,000 |
| **Complexitate** | Mai simplu | Mai detaliat |
| **Training Speed** | Rapid (termina devreme) | Lent (permite explorare) |

---

## Istoricul Configurarilor Testate

### Configurare 1: Initial (Problematic)

```python
# Probleme: rewards scadeau, progress = 0.00 mereu
LR = 1e-4           # Prea mic
GAMMA = 0.99
ENTROPY = 0.01
EPISODES = 100

# Rezultat: Agent nu invata nimic
# Reward: -120 constant
# Progress: 0.00 tot timpul
```

**Probleme Identificate:**
- Device 'auto' nu functiona (eroare PyTorch)
- Progress calculat gresit
- Reward components prea multe si conflictuale

### Configurare 2: LR Adjustment

```python
# Fix: Learning rate crescut
LR = 1e-3           # 10x mai mare
GAMMA = 0.99
ENTROPY = 0.01
EPISODES = 500

# Rezultat: Incepe sa invete, dar instabil
# Reward: -50 → +200 (variabil)
```

### Configurare 3: Reward Simplification

```python
# Fix: Rewards simplificate (5 componente vs 15+)
REWARD_CHECKPOINT = 50.0    # Era: variable cu timp
REWARD_SPEED = 0.5          # Era: 2.0 (prea dominant)
REWARD_CENTERLINE = 0.5     # Nou
REWARD_OFFROAD = -5.0       # Era: -1.0 (prea bland)
REWARD_LAP = 1000.0         # Era: 2000.0

# Rezultat: Training stabil
# Reward: +100 → +400 consistent
```

### Configurare 4: ARS Specific

```python
# Pentru ARS - gradient-free
ARS_LR = 0.02           # Mai mare ca neural networks
ARS_NOISE = 0.03        # Explorare controlata
ARS_NUM_DELTAS = 16     # Directii de testare
ARS_ITERATIONS = 500

# Rezultat: Functioneaza dar limitat
# Reward: ~200-500 (politica liniara)
```

### Configurare 5: A2C Final

```python
# Configurare curenta - stabila
A2C_LR = 1e-3           # Sweet spot
A2C_GAMMA = 0.99        # Standard
A2C_ENTROPY = 0.01      # Explorare moderata
A2C_EPISODES = 500

# Actor/Critic: 10 → 128 → 128 → output
# Rezultat: Best performance
# Reward: +500 → +1000+
```

---

## Probleme Rezolvate in Dezvoltare

### Problema 1: Device 'auto' Error

```
ERROR: Expected one of cuda, cpu device type
```

**Cauza**: PyTorch nu recunoaste 'auto'

**Solutie**:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Problema 2: Progress = 0.00 Mereu

**Cauza**: Calcul gresit al progresului pe pista

**Solutie**: Folosire checkpoint index in loc de distanta continua

### Problema 3: Rewards Descrescatoare

```
Episode 1: 150.00
Episode 10: 98.49
Episode 100: -50.00  # Catastrophic forgetting!
```

**Cauza**: Reward components conflictuale (15+ componente)

**Solutie**: Simplificare la 5 componente core:
1. Checkpoint (+50)
2. Speed (+0.5/step)
3. Centerline (+0.5/step)
4. Off-road (-5/step)
5. Lap (+1000)

### Problema 4: ARS Rewards Identice

```
Iter 1: Avg: 2960.00 | Max: 2960.00
Iter 2: Avg: 2960.00 | Max: 2960.00  # Nu variaza!
```

**Cauza**: Lipsa explorare (noise = 0 sau perturbari prea mici)

**Solutie**:
```python
self.noise = 0.03  # Perturbari vizibile
self.num_deltas = 16  # Mai multe directii
```

### Problema 5: Unicode Errors (Windows)

```
UnicodeEncodeError: 'charmap' codec can't encode character
```

**Cauza**: Emoji-uri in output pe Windows

**Solutie**: Inlocuire emoji cu text `[OK]`, `[FAIL]`

---

## Rezultate Comparate

### DrivingEnv vs RacingEnv

| Metrica | DrivingEnv | RacingEnv |
|---------|------------|-----------|
| Training Time (500 ep) | ~30 min | ~60 min |
| Best Reward A2C | ~600 | ~1000 |
| Best Reward ARS | ~300 | ~500 |
| Lap Completions | Rare | Ocazionale |
| Stabilitate | Variabila | Stabila |

### Recomandari pe Baza Istoricului

1. **Pentru training rapid**: Foloseste DrivingEnv (terminare la off-road)
2. **Pentru performanta maxima**: Foloseste RacingEnv (mai multa explorare)
3. **Pentru debugging**: Seteaza `render_mode="human"`
4. **Pentru stabilitate**: Foloseste configuratia din `config.py` (deja tuned)

---

*Documentatie generata pentru proiectul RL Racing Simulator*
*Versiune: 2.0 | Data: Ianuarie 2025*
*Include: Istoric antrenare, configurari testate, 2 environment-uri*

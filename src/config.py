# src/config.py
class Config:
    # Setari Mediu
    STATE_DIM = 13
    ACTION_DIM = 9  # 9 actiuni in env.py (combinatii accelerare/frana + viraj)
    RENDER_FPS = 60

    # ARS V2 - Improved settings
    ARS_ITERATIONS = 500
    ARS_NUM_DELTAS = 32
    ARS_NUM_BEST_DELTAS = 16
    LR_ARS = 0.03
    ARS_NOISE = 0.025

    # A2C - Improved settings
    A2C_EPISODES = 2000
    LR_A2C = 3e-4
    A2C_GAE_LAMBDA = 0.95
    A2C_ENTROPY_COEF = 0.01

    # PPO
    PPO_EPOCHS = 400
    LR_PPO = 1e-4
    PPO_CLIP_EPS = 0.2
    PPO_GAE_LAMBDA = 0.95

    # Common
    GAMMA = 0.99
    MAX_GRAD_NORM = 0.5

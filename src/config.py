# src/config.py
class Config:
    # Setari Mediu
    STATE_DIM = 13
    ACTION_DIM = 5
    RENDER_FPS = 60

    # ARS 
    ARS_ITERATIONS = 200     
    LR_ARS = 0.02            
    ARS_NUM_DELTAS = 16   

    # A2C 
    A2C_EPISODES = 1000
    LR_A2C = 1e-3
    
    # PPO 
    # de facut
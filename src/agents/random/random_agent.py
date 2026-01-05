import numpy as np
from src.game.actions import Action

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, observation):
        # Returnează o acțiune random (int)
        return self.action_space.sample()
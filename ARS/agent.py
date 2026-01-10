import numpy as np

class ARSAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.02, noise=0.03, num_deltas=16):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.noise = noise
        self.num_deltas = num_deltas
        
        # Politica liniară: Matrice de greutăți (weights)
        self.weights = np.zeros((action_dim, state_dim))

    def select_action(self, state, delta=None, direction=None):
        """
        Calculează acțiunea. Dacă delta este prezent, aplică perturbarea
        pentru explorare în timpul antrenamentului.
        """
        weights = self.weights.copy()
        if delta is not None:
            if direction == "plus":
                weights += delta * self.noise
            else:
                weights -= delta * self.noise
        
        # Înmulțire matriceală: greutăți * observații
        logits = weights.dot(state)
        # Alegem acțiunea cu scorul cel mai mare (argmax)
        return np.argmax(logits)

    def update(self, rollouts, sigma_rewards):
        """
        Actualizează greutățile folosind diferența dintre reward-urile 
        pozitive și negative ale perturbărilor.
        """
        step = np.zeros(self.weights.shape)
        for r_positive, r_negative, delta in rollouts:
            step += (r_positive - r_negative) * delta
        
        self.weights += self.lr / (self.num_deltas * sigma_rewards) * step
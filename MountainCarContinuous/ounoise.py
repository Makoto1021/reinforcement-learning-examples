import numpy as np
import torch
    

class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    """
    def __init__(self, action_space, mu=0.0, theta=0.15, sigma=0.2, sigma_min=0.01, decay_rate=0.995, device=None):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.decay_rate = decay_rate
        self.action_dim = action_space.shape[0]
        self.state = np.ones(self.action_dim) * self.mu
        self.device = device
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        self.sigma = max(self.sigma_min, self.sigma * self.decay_rate)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state, device=self.device, dtype=torch.float32)

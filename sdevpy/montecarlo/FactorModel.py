from abc import ABC, abstractmethod
import numpy as np


class FactorModel(ABC):
    @abstractmethod
    def simulate_step(self, state, dt, dW):
        pass

    @abstractmethod
    def initial_state(self):
        pass


class MultiAssetGBM(FactorModel):
    def __init__(self, S0, mu, sigma):
        self.S0 = np.array(S0)
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        self.n_assets = len(S0)

    def initial_state(self):
        return self.S0.copy()

    def simulate_step(self, state, dt, dW):
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * dW
        return state * np.exp(drift + diffusion)

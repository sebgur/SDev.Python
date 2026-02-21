import numpy as np
from scipy.stats import qmc


class SobolPathGenerator:
    def __init__(self, dim, T, n_steps):
        # self.model = model
        self.n_assets = 7
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.dim = dim
        # self.dim = model.dim
        self.sobol = qmc.Sobol(d=self.dim * n_steps, scramble=True)

    def generate(self, n_paths):
        # S, v = self.model.initial_state(n_paths)

        paths = np.zeros((n_paths, self.n_steps+1, self.n_assets))
        paths[:, 0, :] = 0

        dW = brownian_bridge(n_paths, self.n_steps, self.dim, self.T, self.sobol)

        for t in range(self.n_steps):
            Z = dW[:, :, t] #/ np.sqrt(self.dt)
            # S, v = self.model.step(S, v, self.dt, Z)
            paths[:, t + 1, :] =  paths[:, t + 1, :] + Z#S

        return paths


if __name__ == "__main__":
    print("Hello")
    dim = 2
    T = 5.0
    n_steps = 6
    generator = SobolPathGenerator(dim, T, n_steps)
    paths = generator.generate(3)
    print(paths)

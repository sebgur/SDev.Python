import numpy as np


class PathGenerator:
    def __init__(self, model, corr_engine, T, n_steps):
        self.model = model
        self.corr_engine = corr_engine
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps

    def generate_paths(self, n_paths):
        n_assets = self.model.n_assets
        paths = np.zeros((n_paths, self.n_steps + 1, n_assets))
        state = np.tile(self.model.initial_state(), (n_paths, 1))
        paths[:, 0, :] = state

        for t in range(1, self.n_steps + 1):
            Z = self.corr_engine.correlated_normals(n_paths, n_assets)
            dW = Z * np.sqrt(self.dt)
            state = self.model.simulate_step(state, self.dt, dW)
            paths[:, t, :] = state

        return paths

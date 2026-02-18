import numpy as np


class PathGenerator:
    def __init__(self, model, corr_engine, time_grid, **kwargs):
        seed = kwargs.get("seed", 42)
        np.random.seed(seed)
        self.model = model
        self.corr_engine = corr_engine

        # Time grid
        self.time_grid = time_grid
        # n_steps = len(self.time_grid) - 1
        self.n_steps = len(self.time_grid) - 1

    def generate_paths(self, n_paths):
        n_assets = self.model.n_assets
        paths = np.zeros((n_paths, self.n_steps + 1, n_assets))
        state = np.tile(self.model.initial_state(), (n_paths, 1))
        paths[:, 0, :] = state

        for t_idx in range(1, self.n_steps + 1):
            Z = self.corr_engine.correlated_normals(n_paths, n_assets)
            state = self.model.simulate_step(state, t_idx, Z)
            paths[:, t_idx, :] = state

        return paths

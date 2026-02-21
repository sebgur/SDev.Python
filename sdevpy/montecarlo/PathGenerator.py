import numpy as np


class PathGenerator:
    def __init__(self, model, corr_engine, time_grid, **kwargs):
        seed = kwargs.get("seed", 42)
        np.random.seed(seed)
        self.model = model
        self.corr_engine = corr_engine
        self.time_grid = time_grid
        self.n_steps = len(self.time_grid) - 1

    def generate_paths(self, n_paths):
        # TODO: this is where we call the BM path construction
        # But don't we need to reverse the dimensions?
        # And how do we apply the correlation, which is in the dimensions
        # of the factors, to be applied to a slice of increments at a
        # specific time index?
        n_assets = self.model.n_assets
        paths = np.zeros((n_paths, self.n_steps + 1, n_assets))
        state = np.tile(self.model.initial_state(), (n_paths, 1))
        paths[:, 0, :] = state

        # Path construction along the time direction
        for t_idx in range(1, self.n_steps + 1):
            Z = self.corr_engine.correlate_normals(n_paths, n_assets)
            state = self.model.simulate_step(state, t_idx, Z)
            paths[:, t_idx, :] = state

        return paths

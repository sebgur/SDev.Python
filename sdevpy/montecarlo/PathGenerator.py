import numpy as np
from sdevpy.maths.rand.pathconstruction import get_path_builder


class PathGenerator:
    def __init__(self, model, time_grid, **kwargs):
        self.model = model
        self.time_grid = time_grid
        self.n_steps = len(self.time_grid) - 1
        self.n_factors = self.model.n_factors
        self.path_builder = get_path_builder(time_grid, self.n_factors, **kwargs)

    # def generate_paths(self, n_paths):
    #     n_assets = self.model.n_assets
    #     paths = np.zeros((n_paths, self.n_steps + 1, n_assets))
    #     state = np.tile(self.model.initial_state(), (n_paths, 1))
    #     paths[:, 0, :] = state

    #     # Path construction along the time direction
    #     for t_idx in range(1, self.n_steps + 1):
    #         Z = self.corr_engine.correlate_normals(n_paths, n_assets)
    #         state = self.model.simulate_step(state, t_idx, Z)
    #         paths[:, t_idx, :] = state

    #     return paths

    def generate_paths(self, n_paths):
        paths = np.zeros((n_paths, self.n_steps + 1, self.n_factors))
        state = np.tile(self.model.initial_state(), (n_paths, 1))
        paths[:, 0, :] = state

        # Generate Brownian path
        bm_paths = self.path_builder.build(n_paths)

        # Construct underlying paths along the time direction
        for t_idx in range(1, self.n_steps + 1):
            dW = bm_paths[:, t_idx - 1, :] # Increments for this time step
            state = self.model.evolve_state(state, t_idx, dW)
            paths[:, t_idx, :] = state

        return paths


if __name__ == "__main__":
    print("Hello")

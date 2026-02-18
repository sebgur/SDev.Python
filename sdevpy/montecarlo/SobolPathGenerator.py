

class SobolHestonPathGenerator:
    def __init__(self, model, T, n_steps):
        self.model = model
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.dim = model.dim
        self.sobol = qmc.Sobol(d=self.dim * n_steps, scramble=True)

    def generate(self, n_paths):
        S, v = self.model.initial_state(n_paths)

        paths = np.zeros((n_paths, self.n_steps+1, self.model.n_assets))
        paths[:, 0, :] = S

        dW = brownian_bridge(n_paths, self.n_steps, self.dim, self.T, self.sobol)

        for t in range(self.n_steps):
            Z = dW[:, :, t] / np.sqrt(self.dt)
            S, v = self.model.step(S, v, self.dt, Z)
            paths[:, t+1, :] = S

        return paths

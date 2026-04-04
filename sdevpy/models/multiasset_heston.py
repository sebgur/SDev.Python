import numpy as np


class MultiAssetHeston:
    def __init__(self, s0, v0, r, q, kappa, theta, xi, corr_matrix):
        self.s0 = np.array(s0)
        self.v0 = np.array(v0)
        self.r = r
        self.q = np.array(q)
        self.kappa = np.array(kappa)
        self.theta = np.array(theta)
        self.xi = np.array(xi)
        self.n_assets = len(s0)
        self.dim = 2 * self.n_assets
        self.L = np.linalg.cholesky(corr_matrix)

    def initial_state(self, n_paths):
        s = np.tile(self.S0, (n_paths, 1))
        v = np.tile(self.v0, (n_paths, 1))
        return s, v

    def step(self, s, v, dt, z):
        # correlate
        z_corr = z @ self.L.T
        zs = z_corr[:, :self.n_assets]
        zv = z_corr[:, self.n_assets:]

        # variance - full truncation Euler
        v = np.maximum(v, 0)
        dv = self.kappa * (self.theta - v) * dt + self.xi * np.sqrt(v * dt) * zv
        v_next = np.maximum(v + dv, 0)

        # price
        drift = (self.r - self.q - 0.5 * v) * dt
        diffusion = np.sqrt(v * dt) * zs
        s_next = s * np.exp(drift + diffusion)
        return s_next, v_next

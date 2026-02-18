import numpy as np


class CorrelationEngine:
    def __init__(self, corr_matrix):
        self.L = np.linalg.cholesky(corr_matrix)

    def correlated_normals(self, n_paths, n_factors):
        Z = np.random.normal(size=(n_paths, n_factors))
        return Z @ self.L.T

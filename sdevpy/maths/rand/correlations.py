import numpy as np


class CorrelationEngine:
    def __init__(self, corr_matrix):
        self.L = np.linalg.cholesky(corr_matrix)

    def correlate_normals(self, n_paths, n_factors):
        Z = np.random.normal(size=(n_paths, n_factors))
        return Z @ self.L.T

    def correlate_paths(self, paths):
        return paths @ self.L.T


if __name__ == "__main__":
    # Define correlation matrix
    corr = np.asarray([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]])

    # Decompose
    chol = np.linalg.cholesky(corr)
    print(chol)
    print(chol.T)
    print(chol @ chol.T) # Check consistency of decomposition
    n_factors = len(corr)
    print(f"Number of factors: {n_factors}")

    # Generate independent paths
    n_paths = 100000
    n_timesteps = 25
    ind_gaussians = np.random.normal(size=(n_paths, n_timesteps, n_factors))

    # Correlate paths using matrix multiplication
    corr_gaussians = ind_gaussians @ chol.T

    # Check correlations
    rho12 = np.corrcoef(corr_gaussians[:, 0, 0], corr_gaussians[:, 0, 1])[0, 1]
    rho13 = np.corrcoef(corr_gaussians[:, 0, 0], corr_gaussians[:, 0, 2])[0, 1]
    rho23 = np.corrcoef(corr_gaussians[:, 0, 1], corr_gaussians[:, 0, 2])[0, 1]
    check = np.corrcoef(corr_gaussians[:, 0, 0], corr_gaussians[:, 1, 1])[0, 1]
    print(rho12)
    print(rho13)
    print(rho23)
    print(check)

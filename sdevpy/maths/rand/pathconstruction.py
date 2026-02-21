import numpy as np
from abc import ABC, abstractmethod
from sdevpy.maths.rand.rng import get_rng
from sdevpy.maths.rand.correlations import CorrelationEngine


def get_path_builder(time_grid, n_factors=1, **kwargs):
    """ Create a Brownian path constructor """
    # Get random number generator
    n_steps = len(time_grid) - 1
    rng = get_rng(dim=n_steps * n_factors, **kwargs)

    # Get path constructor
    constr_type = kwargs.get('constr_type', 'incremental')
    path_builder = None
    match constr_type.lower():
        case 'incremental': path_builder = IncrementalPathBuilder(time_grid, n_factors, rng)
        case 'brownianbridge': path_builder = BrownianBridgePathBuilder(time_grid, n_factors, rng)
        case _: raise TypeError(f"Invalid path construction type: {constr_type}")

    # Correlations
    corr_matrix = kwargs.get('corr_matrix', None)
    path_builder.set_correlations(corr_matrix)

    return path_builder


def brownianbridge(n_paths, time_grid, n_factors, rng):
    n_steps = len(time_grid) - 1
    T = time_grid[-1]

    # Draw gaussians
    Z = rng.normal(n_paths)

    # Allocate dimensions: split (factor1, factor2, ...) into (factor1 x factor x ...)
    Z = Z.reshape(n_paths, n_factors, n_steps)

    # Build paths
    W = np.zeros((n_paths, n_factors, n_steps + 1)) # Because we use the first point at 0
    for d in range(n_factors):
        # First assign W(T)
        W[:, d, -1] = np.sqrt(T) * Z[:, d, 0]

        # Recursive midpoint fill
        intervals = [(0, n_steps)]

        z_index = 1
        while intervals:
            start, end = intervals.pop(0)
            mid = (start + end) // 2
            if mid == start or mid == end:
                continue

            t_start = time_grid[start]
            t_mid = time_grid[mid]
            t_end = time_grid[end]

            mean = ((t_mid - t_start) * W[:, d, end] + (t_end - t_mid) * W[:, d, start]) / (t_end - t_start)
            var = (t_mid - t_start) * (t_end - t_mid) / (t_end - t_start)

            W[:, d, mid] = mean + np.sqrt(var) * Z[:, d, z_index]

            z_index += 1

            intervals.append((start, mid))
            intervals.append((mid, end))

    # Convert to increments
    dW = np.diff(W, axis=2)
    return dW


def brownianbridge2(n_paths, time_grid, n_factors, rng):
    n_steps = len(time_grid) - 1
    T = time_grid[-1]

    # Draw gaussians
    Z = rng.normal(n_paths)

    # Allocate dimensions: split (factor1, factor2, ...) into (factor1 x factor x ...)
    Z = Z.reshape(n_paths, n_steps, n_factors)

    # Build paths
    W = np.zeros((n_paths, n_steps + 1, n_factors)) # Because we use the first point at 0
    for d in range(n_factors):
        # First assign W(T)
        W[:, -1, d] = np.sqrt(T) * Z[:, 0, d]

        # Recursive midpoint fill
        intervals = [(0, n_steps)]

        z_index = 1
        while intervals:
            start, end = intervals.pop(0)
            mid = (start + end) // 2
            if mid == start or mid == end:
                continue

            t_start = time_grid[start]
            t_mid = time_grid[mid]
            t_end = time_grid[end]

            mean = ((t_mid - t_start) * W[:, end, d] + (t_end - t_mid) * W[:, start, d]) / (t_end - t_start)
            var = (t_mid - t_start) * (t_end - t_mid) / (t_end - t_start)

            W[:, mid, d] = mean + np.sqrt(var) * Z[:, z_index, d]

            z_index += 1

            intervals.append((start, mid))
            intervals.append((mid, end))

    # Convert to increments
    dW = np.diff(W, axis=1)
    return dW


class PathBuilder(ABC):
    def __init__(self, time_grid, n_factors, rng):
        self.time_grid = time_grid
        self.n_factors = n_factors
        self.rng = rng
        self.corr_engine = None

    @abstractmethod
    def build_independent(self, n_paths):
        pass

    def build(self, n_paths):
        ind_paths = self.build_independent2(n_paths)
        if self.corr_engine is None:
            return ind_paths
        else:
            return self.corr_engine.correlate_paths(ind_paths)

    def set_correlations(self, corr_matrix):
        if corr_matrix is not None:
            self.corr_engine = CorrelationEngine(corr_matrix)


class IncrementalPathBuilder(PathBuilder):
    def build_independent(self, n_paths):
        n_steps = len(self.time_grid) - 1

        # Draw gaussians
        Z = self.rng.normal(n_paths)

        # Allocate dimensions: split (factor1, factor2, ...) into (factor1 x factor x ...)
        Z = Z.reshape(n_paths, self.n_factors, n_steps)

        # Build paths
        dt = np.diff(self.time_grid)
        dW = Z * np.sqrt(dt)
        return dW

    def build_independent2(self, n_paths):
        n_steps = len(self.time_grid) - 1

        # Draw gaussians
        Z = self.rng.normal(n_paths)

        # Allocate dimensions: split (factor1, factor2, ...) into (factor1 x factor x ...)
        Z = Z.reshape(n_paths, n_steps, self.n_factors)

        # Build paths
        dt = np.diff(self.time_grid)
        dt = dt.reshape(1, n_steps, 1)
        dW = Z * np.sqrt(dt)
        return dW


class BrownianBridgePathBuilder(PathBuilder):
    def build_independent(self, n_paths):
        return brownianbridge(n_paths, self.time_grid, self.n_factors, self.rng)

    def build_independent2(self, n_paths):
        return brownianbridge2(n_paths, self.time_grid, self.n_factors, self.rng)


if __name__ == "__main__":
    # We find that although the generation of the random numbers is dominant at runtime,
    # the Brownian bridge construction is a significant runtime drag compared to the
    # incremental construction.
    # Based on the study in https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1951886
    # the scrambling was very efficient to remove Sobol's bias while being much simpler
    # to implement.
    # Here we find that the scrambling (assuming it has the same efficiency as the version
    # studied in the reference above) has a very minor effect on the runtime of the
    # Sobol sequence generation.
    # Therefore, from a runtime perspective, it seems more advantageous to run
    # Sobol with scrambling in incremental construction thatn Sobol without scrambling
    # in the Brownian bridge construction.
    from sdevpy.maths.rand import rng
    from sdevpy.tools import timer

    # Make time grid
    n_steps = 5
    t = 5.0
    time_grid = np.linspace(0, t, n_steps + 1)

    # Generate path
    n_paths = 10000
    n_factors = 3
    # constr_type = 'brownianbridge'
    constr_type = 'incremental'
    print(f"Number of paths: {n_paths}")
    print(f"Number of time steps: {n_steps}")
    print(f"Number of factors: {n_factors}")
    print(f"Constructor type: {constr_type}")

    path1 = get_path_builder(time_grid, n_factors, constr_type=constr_type, rng_type='Sobol')
    dw1 = path1.build_independent(n_paths)
    print(f"Brownian increment path shape: {dw1.shape}")
    # print(dw)

    path2 = get_path_builder(time_grid, n_factors, constr_type=constr_type, rng_type='Sobol')

    # Correlation matrix
    corr = np.asarray([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]])
    path2.set_correlations(corr)
    dw2 = path2.build(n_paths)
    print(f"Brownian increment path shape: {dw2.shape}")
    # print(dw)

    # Check path properties
    # We expect covariances to be equal to the corresponding time step coefficients times
    # the correlations, which should be 0 if the time steps are different.
    factor1, factor2 = 1, 2
    t_idx1, t_idx2 = 1, 1
    s1 = dw2[:, t_idx1, factor1]
    s2 = dw2[:, t_idx2, factor2]
    rho11 = np.cov(s1, s1)[0, 1]
    rho12 = np.cov(s1, s2)[0, 1]
    rho22 = np.cov(s2, s2)[0, 1]
    dt11 = time_grid[t_idx1 + 1] - time_grid[t_idx1]
    dt12 = np.sqrt((time_grid[t_idx1 + 1] - time_grid[t_idx1]) * (time_grid[t_idx2 + 1] - time_grid[t_idx2]))
    dt22 = time_grid[t_idx2 + 1] - time_grid[t_idx2]
    print(rho11 / dt11)
    print(rho12 / dt12)
    print(rho22 / dt22)

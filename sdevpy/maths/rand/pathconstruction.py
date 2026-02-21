import numpy as np
from abc import ABC, abstractmethod


def get_path_constructor(n_factors, time_grid, rng, constr_type='brownianbridge'):
    """ Create a Brownian path constructor """
    match constr_type.lower():
        case 'incremental': return IncrementalPathConstructor(n_factors, time_grid, rng)
        case 'brownianbridge': return BrownianBridgePathConstructor(n_factors, time_grid, rng)
        case _: raise TypeError(f"Invalid path construction type: {constr_type}")


def brownianbridge(n_paths, n_factors, time_grid, rng):
    n_steps = len(time_grid) - 1
    T = time_grid[-1]

    st = timer.Stopwatch("Sobol")

    # Draw gaussians
    Z = rng.normal(n_paths)

    # Allocate dimensions: split (factor1, factor2, ...) into (factor1 x factor x ...)
    Z = Z.reshape(n_paths, n_factors, n_steps)

    st.stop()
    bp = timer.Stopwatch("Brownian path")

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

    bp.stop()
    st.print()
    bp.print()

    return dW


class PathConstructor(ABC):
    def __init__(self, n_factors, time_grid, rng):
        self.n_factors = n_factors
        self.time_grid = time_grid
        self.rng = rng

    @abstractmethod
    def build(self, n_paths):
        pass


class IncrementalPathConstructor(PathConstructor):
    def build(self, n_paths):
        n_steps = len(time_grid) - 1

        st = timer.Stopwatch("Sobol")

        # Draw gaussians
        Z = rng.normal(n_paths)

        # Allocate dimensions: split (factor1, factor2, ...) into (factor1 x factor x ...)
        Z = Z.reshape(n_paths, n_factors, n_steps)

        st.stop()
        bp = timer.Stopwatch("Brownian path")

        # Build paths
        dt = np.diff(time_grid)
        dW = Z * np.sqrt(dt)

        bp.stop()
        st.print()
        bp.print()

        return dW


class BrownianBridgePathConstructor(PathConstructor):
    def build(self, n_paths):
        return brownianbridge(n_paths, self.n_factors, self.time_grid, self.rng)


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
    from sdevpy.maths.rand import sobol
    from sdevpy.tools import timer

    # Make time grid
    n_steps = 3
    t = 5.0
    time_grid = np.linspace(0, t, n_steps + 1)

    # Generate path
    n_paths = 7000000
    n_factors = 5
    # constr_type = 'brownianbridge'
    constr_type = 'incremental'
    print(f"Number of paths: {n_paths}")
    print(f"Number of factors: {n_factors}")
    print(f"Number of time steps: {n_steps}")
    print(f"Constructor type: {constr_type}")

    n_steps = len(time_grid) - 1
    rng = sobol.Sobol(n_factors * n_steps, scramble=True)
    path_constructor = get_path_constructor(n_factors, time_grid, rng, constr_type=constr_type)
    dw = path_constructor.build(n_paths)

    print(f"Brownian increment path shape: {dw.shape}")
    # print(dw)

from montecarlo.Simulator import Simulator
import numpy as np


class BlackSimulator(Simulator):
    def __init__(self, repo_rate, div_rate, expiry, debug=False):
        self.repo_rate = repo_rate
        self.div_rate = div_rate
        self.expiry = expiry
        self.debug = debug

    def build_paths(self, init_spot, init_vol, rng):
        data_shape = init_spot.shape
        if init_vol.shape != data_shape:
            raise RuntimeError("Incompatible sizes between spots and vols")

        num_samples = data_shape[0]
        num_underlyings = data_shape[1]

        # Calculate deterministic quantities
        t = self.expiry
        sqrt_t = np.sqrt(t)
        fwd_ratio = np.ndarray(shape=num_underlyings)
        for j in range(num_underlyings):
            fwd_ratio[j] = np.exp((self.repo_rate - self.div_rate) * t)

        # Generate gaussians
        dg = rng.normal(0.0, 1.0, data_shape)
        if self.debug:
            print("Gaussians")
            print(dg)

        # Calculate future spots
        future_spot = np.ndarray(data_shape)
        for i in range(num_samples):
            for j in range(num_underlyings):
                fwd = init_spot[i, j] * fwd_ratio[j]
                stdev = init_vol[i, j] * sqrt_t
                future_spot[i, j] = fwd * np.exp(-0.5 * stdev * stdev + stdev * dg[i, j])

        return future_spot

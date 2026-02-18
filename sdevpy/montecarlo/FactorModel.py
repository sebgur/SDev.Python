from abc import ABC, abstractmethod
import numpy as np


class FactorModel(ABC):
    @abstractmethod
    def simulate_step(self, state, t_idx, Z):
        pass

    @abstractmethod
    def initial_state(self):
        pass


class MultiAssetGBM(FactorModel):
    def __init__(self, spot, lv, fwd_curve, time_grid):
        self.spot = spot
        self.fwd_curve = fwd_curve
        self.lv = lv
        self.time_grid = time_grid
        self.n_assets = len(self.spot) # Used by PathGenerator

        # Cache forwards
        fwd_grid = []
        for t in time_grid:
            fwd_grid.append(np.asarray([f(t) for f in fwd_curve]))
        self.fwd_grid = np.asarray(fwd_grid)

        # TODO: Here would be a good place to retrieve the local vol
        # functions at each time step.
        # Also we need to check how the lv.value() works on
        # the state variable with its shape.


    def initial_state(self):
        return self.spot.copy()

    def simulate_step(self, state, t_idx, Z):
        fs = self.fwd_grid[t_idx - 1]
        fe = self.fwd_grid[t_idx]
        ts = self.time_grid[t_idx - 1]
        te = self.time_grid[t_idx]

        # Calculate the log-moneyness to evaluate the local vol
        logms = np.log(state / fs)
        vol = self.lv

        # Now evolve
        dt = te - ts
        dW = Z * np.sqrt(dt)
        return fe * np.exp(logms - 0.5 * vol**2 * dt + vol * dW)

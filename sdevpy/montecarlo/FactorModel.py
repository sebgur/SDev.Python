from abc import ABC, abstractmethod
import numpy as np


class FactorModel(ABC):
    @abstractmethod
    def evolve_state(self, state, t_idx, dW):
        pass

    # @abstractmethod
    # def simulate_step(self, state, t_idx, Z):
    #     pass

    @abstractmethod
    def initial_state(self):
        pass


class MultiAssetGBM(FactorModel):
    def __init__(self, spot, sigma, lv, fwd_curve, time_grid, **kwargs):
        self.spot = spot
        self.fwd_curve = fwd_curve
        self.sigma = sigma
        self.lv = lv
        self.time_grid = time_grid
        self.n_factors = len(self.spot) # Used by PathGenerator
        self.use_lv = kwargs.get('use_lv', False)

        # Cache forwards
        fwd_grid = []
        for t in time_grid:
            fwd_grid.append(np.asarray([f(t) for f in fwd_curve]))
        self.fwd_grid = np.asarray(fwd_grid)

    def initial_state(self):
        return self.spot.copy()

    # def simulate_step(self, state, t_idx, Z):
    #     fs = self.fwd_grid[t_idx - 1]
    #     fe = self.fwd_grid[t_idx]
    #     ts = self.time_grid[t_idx - 1]
    #     te = self.time_grid[t_idx]

    #     # Calculate the log-moneyness to evaluate the local vol
    #     logms = np.log(state / fs)
    #     vol = self.lv

    #     # Now evolve
    #     dt = te - ts
    #     dW = Z * np.sqrt(dt)
    #     return fe * np.exp(logms - 0.5 * vol**2 * dt + vol * dW)

    def evolve_state(self, state, t_idx, dW):
        fs = self.fwd_grid[t_idx - 1]
        fe = self.fwd_grid[t_idx]
        ts = self.time_grid[t_idx - 1]
        te = self.time_grid[t_idx]

        # Calculate the log-moneyness to evaluate the local vol
        logms = np.log(state / fs)
        # Vols
        if self.use_lv == False: # Constant vol
            vol = self.sigma
        else: # Local vol
            lvols = np.asarray([self.lv[i].value(ts, logms[:, i]) for i in range(self.n_factors)])
            lvols = lvols.T
            vol = lvols

        # Now evolve
        dt = te - ts
        return fe * np.exp(logms - 0.5 * vol**2 * dt + vol * dW)


if __name__ == "__main__":
    import datetime as dt
    from sdevpy.models import localvol_factory as lvf
    path = np.asarray([[-0.5, 0.1, 0.5], [-0.1, 0.15, 0.4], [-0.0001, 0.05, 0.25], [0.12, 0.07, 0.18]])
    print(f"Path: {path.shape}")

    # Get LV
    name = "CalibIndex" # "MyIndex"
    valdate = dt.datetime(2025, 12, 15)
    folder = lvf.test_data_folder()

    # Load
    store_date = valdate
    new_expiries = None
    lv = lvf.load_lv_from_folder(new_expiries, store_date, name, folder)
    lv.name = name
    lv.valdate = valdate

    # View
    expiries = lv.t_grid
    n_expiries = len(expiries)
    exp_idx = n_expiries - 1
    lvs = [lv, lv, lv]
    lvols = np.asarray([(i+1) * lvs[i].value(expiries[exp_idx], path[:, i]) for i in range(3)])
    # lvols = lv.value(expiries[exp_idx], path[:, 0])
    print(lvols)
    print(f"LVs: {lvols.shape}")
    lvols = lvols.T
    print(lvols)
    print(f"LVs: {lvols.shape}")

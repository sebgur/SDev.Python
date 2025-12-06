from abc import ABC, abstractmethod

########## ToDo (calibration) #################################################
# * Implement local vol design
# * Use seaborn to represent diffs between IV and LV prices on quoted pillars
# * Add 1d solving to ATM only, to do live and Vega with smile solving less often.
# * During the warmup in the time direction, we can allocate
#   on each time slice a local vol functional form that is only a function
#   of the spot. This would be a generalized version of the storage
#   of the time interpolation indices for an interpolated surface.
# * A spot parametric local vol would be spot-parametric on predefined
#   time slices, and would for instance take the same parametric form
#   over forward time intervals.
# * We could resolve forward by taking the previous parametric form as
#   starting point.
# * We could use SVI as base and if not enough point, fit only to reduced
#   set of free parameters, the other ones defaulting to good solver starting points.
# * For the (backward) pricing PDE, also allow the standard case of a fully interpolated
#   matrix, using cubic splines with flat extrapolation on both ends and arriving flat
#   first derivative.
# * To check the quality of the calibration, start by comparing against same forward
#   PDE as used in calibration, and then check against backward PDE.

class InterpolatedParametricLocalVol(LocalVol):
    def __init__(self, t_grid, param_grid):
        self.t_grid = t_grid
        # Strip parameters
        n_times = t_grid.shape[0]
        shape = param_grid.shape
        if shape[0] != n_times:
            raise RuntimeError("Incorrect sizes between time grid and parameters")

        # self.alnv_grid = parameter_grid[:, 0]
        # self.b_grid = parameter_grid[:, 1]
        # self.rho_grid = parameter_grid[:, 2]
        # self.m_grid = parameter_grid[:, 3]
        # self.sigma_grid = parameter_grid[:, 4]
        for t_idx, t in enumerate(self.t_grid):
            self.check_params(t, param_grid[t_idx])

    def check_params(t, params):
        return 0

    def update_params(t_idx, new_params):
        return 0



class LocalVol(ABC):
    """ Base class for local vols """
    @abstractmethod
    def value(self, t, x):
        """ Where x is the log-moneyness """
        pass

    @abstractmethod
    def section(self, t):
        """ Get a section at time t, i.e. a function of the log-moneyness x"""
        pass


if __name__ == "__main__":
    print("Hello")

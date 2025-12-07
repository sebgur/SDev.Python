from abc import ABC, abstractmethod
from sdevpy.models.svi import *

########## ToDo (LocalVol) #################################################
# * Implement/Finish testing the time lookup with tolerance
# * Include it in the definition of the localVol
# * Initialize LV
# * Create objective function, constraints, with pre-calculation of weighted
#   payoff that doesn't depend on vols, define it as a generic payoff in a
#   payoff class.
# * Define the payoff class's core algebraic operations for Add, Multiply, etc.
# * Refresh optimizer implementation, get definition/control of stopping
#   criteria.

########## ToDo (calibration) #################################################
# * Implement calibration by sections
# * Use seaborn to represent diffs between IV and LV prices on quoted pillars
# * Add 1d solving to ATM only, to do live and Vega with smile solving less often.
# * During the warmup in the time direction, we can allocate
#   on each time slice a local vol functional form that is only a function
#   of the spot. This would be a generalized version of the storage
#   of the time interpolation indices for an interpolated surface.
# * We could resolve forward by taking the previous parametric form as
#   starting point.
# * For the (backward) pricing PDE, also allow the standard case of a fully interpolated
#   matrix, using cubic splines with flat extrapolation on both ends and arriving flat
#   first derivative.
# * To check the quality of the calibration, start by comparing against same forward
#   PDE as used in calibration. Define a simple method that calculates the whole
#   surface. Then implement and check against backward PDE.
# * Make notebook that illustrates the whole flow.
# * Introduce unit testing. Cleanup package, upload to pypi.
# * Make Colab, post.

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


class InterpolatedParamLocalVol(LocalVol):
    def __init__(self, t_grid, section_grid):
        self.t_grid = t_grid
        self.section_grid = section_grid
        # Size consistency
        if len(section_grid) != t_grid.shape[0]:
            raise RuntimeError("Incorrect sizes between time grid and section grid")

    def value(self, t, x):
        return 0

    def section(self, t):
        return 0

    def check_params(self, t_idx):
        return self.section_grid[t_idx].check_params()

    def update_params(self, t_idx, new_params):
        self.section_grid[t_idx].update_params(new_params)


if __name__ == "__main__":
    t_grid = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    section_grid = [SviSection() for i in range(t_grid.shape[0])]
    lv = InterpolatedParamLocalVol(t_grid, section_grid)
    print("Hello")

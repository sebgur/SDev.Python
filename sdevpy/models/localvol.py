from abc import ABC, abstractmethod
from sdevpy.models.svi import *
from sdevpy.tools import algos


class LocalVol(ABC):
    """ Base class for local vols """
    @abstractmethod
    def value(self, t, logm):
        pass

    @abstractmethod
    def section(self, t):
        """ Get a section at time t, i.e. a function of the log-moneyness log_m"""
        pass


class InterpolatedParamLocalVol(LocalVol):
    def __init__(self, t_grid, section_grid):
        self.t_grid = t_grid
        self.section_grid = section_grid
        # Size consistency
        if len(section_grid) != t_grid.shape[0]:
            raise RuntimeError("Incorrect sizes between time grid and section grid")

    def value(self, t, logm):
        t_idx = algos.upper_bound(t_grid, t)
        return section_grid[t_idx].value(t, logm)

    def section(self, t):
        return 0

    def check_params(self, t_idx):
        return self.section_grid[t_idx].check_params()

    def update_params(self, t_idx, new_params):
        self.section_grid[t_idx].update_params(new_params)


if __name__ == "__main__":
    t_grid = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    alnv = 0.25
    a = alnv**2 * np.sqrt(1.0) # a > 0
    b = 0.2 # b > 0
    rho = 0.0 # -1 < rho < 1
    m = 0.5 # No constraints
    sigma = 0.25 # > 0
    params = [a, b, rho, m, sigma]

    section_grid = [SviSection() for i in range(t_grid.shape[0])]
    lv = InterpolatedParamLocalVol(t_grid, section_grid)

    # Initialize parameters
    for t_idx in range(len(t_grid)):
        lv.update_params(t_idx, params)

    lv_t = 0.75
    lv_logm = 0.0
    lvol = lv.value(lv_t, lv_logm)
    print(f"T/LOG-M/LV: {lv_t}/{lv_logm}/{lvol}")

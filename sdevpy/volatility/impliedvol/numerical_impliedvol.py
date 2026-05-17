import numpy as np
import numpy.typing as npt
from sdevpy.volatility.impliedvol.impliedvol import ImpliedVol
from sdevpy.volatility.localvol.localvol import LocalVol
from sdevpy.volatility.impliedvol.optionsurface import OptionQuoteType
from sdevpy.pde import forwardpde as fpde
from sdevpy.pde.pdeschemes import PdeConfig
from sdevpy.utilities.tools import isequal


DFLT_PDE_CONFIG = PdeConfig()


########### TODO ################################
# * Write some tests for build_step_grid
# * Test in Jupyter
# * Pass simpler than PdeConfig by estimating mesh_vol using LV?


class NumericalImpliedVol(ImpliedVol):
    def __init__(self, lv: LocalVol, pde_config: PdeConfig=DFLT_PDE_CONFIG, **kwargs):
        super().__init__()
        self.lv = lv
        self.calculate_type = OptionQuoteType.ForwardPremium
        self.pde_config = pde_config

    def calculate(self, t: float, k: npt.ArrayLike, is_call: bool, f: float) -> npt.ArrayLike:
        """ Calculate the forward prices using a numerical method (typically forward PDE) """
        start_time = 1.0 / 365.0
        if t < start_time:
            raise ValueError(f"Numerical method not supported for use before 1D, used with t = {t}")

        # Create a step grid to run the density steps
        step_grid = self.build_step_grid(t)
        print(f"Maturity: {t}")
        print(f"Step grid: {step_grid}")
        if len(step_grid) < 1:
            raise ValueError("Invalid step grid for PDE")

        if np.abs(step_grid[-1] - t) > self.time_epsilon:
            raise ValueError(f"Invalid PDE time step grid, last point not equal to maturity: {step_grid[-1]}/{t}")

        dens_report = fpde.calculate_densities(step_grid, self.lv, self.pde_config)[-1]
        dens_t, dens_p, dens_x = dens_report['end_time'], dens_report['p_grid'], dens_report['x_grid']
        if not isequal(dens_t, t):
            raise ValueError(f"Unexpected time for final density not equal to maturity: {dens_t}/{t}")

        # # Build first spot grid
        # old_x, old_dx, old_spot_idx = fpde.build_spotgrid(step_grid[0], self.pde_config)

        # # Initialize density
        # old_p = fpde.lognormal_density(old_x, start_time, self.pde_config.mesh_vol)

        # # Evolve density until maturity
        # for i in range(len(step_grid)):
        #     ts = (start_time if i == 0 else step_grid[i - 1])
        #     te = step_grid[i]
        #     print(f"Evolving PDE from {ts} to {te}")

        #     # Evolve density
        #     t_grid = timegrids.build_timegrid(ts, te, self.pde_config)
        #     old_x, old_dx, old_p = fpde.density_step(old_p, old_x, old_dx, t_grid, self.lv.value, self.pde_config)

        # Calculate prices
        prices = []
        spot = f * np.exp(dens_x)
        for strike in k:
            payoff = np.max(spot - strike, 0.0) if is_call else np.max(strike - spot, 0.0)
            prices.append(fpde.expectation(payoff, dens_p, dens_x))

        return np.asarray(prices)

    def dump_data(self) -> dict:
        """ Dump to dictionary """
        result = {'lv': self.lv.dump_data()}
        return result

    @staticmethod
    def build_step_grid(t: float) -> list[float]:
        """ Construct PDE time step grid with minimum granularity """
        if t < 0.0:
            raise ValueError(f"Negative time requested in PDE time step grid building: {t}")

        coarse_grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        term_tol = 1.0 / 52.0
        long_term_size = 1.0 # In years
        step_grid = [point for point in coarse_grid if point < t]
        current_t = step_grid[-1] + long_term_size
        cut_off_years = 100.0
        while current_t < t:
            step_grid.append(current_t)
            current_t += long_term_size
            if current_t > cut_off_years:
                raise ValueError("Unexpected large step grid built for PDE")

        if step_grid[-1] > t - term_tol:
            step_grid[-1] = t
        else:
            step_grid.append(t)

        return step_grid

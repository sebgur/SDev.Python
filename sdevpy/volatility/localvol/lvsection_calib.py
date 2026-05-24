import datetime as dt
import numpy as np
import numpy.typing as npt
from sdevpy.utilities import timegrids, dates
from sdevpy.utilities.tools import isequal
from sdevpy.volatility.localvol import localvol_factory as lvf
from sdevpy.volatility.localvol.localvol import TimeInterpolatedLocalVol
from sdevpy.volatility.impliedvol.optionsurface import calibration_targets
from sdevpy.pde.pdeschemes import PdeConfig
from sdevpy.pde import forwardpde as fpde
from sdevpy.analytics import black
from sdevpy.maths import metrics, constants
from sdevpy.maths.optimization import create_optimizer
from sdevpy.market import eqvolsurface as vsurf
from sdevpy.market.eqforward import get_forward_curves
from sdevpy.instruments.constants import OptionType, string_to_optiontype


def calibrate_lv_bysections(valdate: dt.datetime, name: str, config: dict, **kwargs) -> dict:
    """ Calibrate InterpolatedParamLocalVol type to market data """
    # Arguments
    verbose = kwargs.get('verbose', False)
    disp_opt = kwargs.get('disp_opt', False)
    calc_pde_vols = kwargs.get('calc_pde_vols', False)

    # Retrieve forward curve
    fwd_curve = get_forward_curves([name], valdate)[0]

    # Retrieve target market option data
    file = vsurf.data_file(name, valdate)
    surface_data = vsurf.eqvolsurfacedata_from_file(file)
    expiries = surface_data.expiries
    fwds = fwd_curve.value(expiries)
    strike_surface = surface_data.get_strikes(fwd_curve=fwd_curve, to_type='absolute')
    vol_surface = surface_data.vols

    # Set calibration time grid
    expiry_grid = np.array([timegrids.model_time(valdate, expiry) for expiry in expiries])

    # Set calibration targets
    option_type = 'straddle'
    cf_price_surface, ftols = calibration_targets(expiry_grid, fwds, strike_surface, vol_surface,
                                                  option_type=option_type)

    # Initial LV: either from scratch or from existing
    lv_t_grid = [0.0] # LV time grid
    lv_t_grid.extend(expiry_grid[:-1])
    lv = lvf.load_param_lv(valdate, name, t_grid=lv_t_grid, folder=config.get('lv_folder', None))
    lv.name, lv.valdate, lv.snapdate = name, valdate, valdate
    # print(f"IV time grid: {expiry_grid}")
    # print(f"LV time grid: {lv.t_grid}")

    # Set forward PDE
    pde_config = fpde.PdeConfig(n_timesteps=config['pde_timesteps'], n_meshes=config['pde_spotsteps'],
                                scheme='rannacher', rescale_x=True, rescale_p=True,
                                shift_forward=False)

    # Set objective
    obj_builder = LvObjectiveBuilder(lv, expiry_grid, fwds, strike_surface, cf_price_surface, pde_config)
    objective = obj_builder.objective

    # Optimizer settings
    method = config['optimizer']
    tol = config['tol']

    # Initialize PDE
    # old_x, old_dx, old_p = obj_builder.initialize()
    old_x, old_dx, old_p = None, None, None

    if verbose:
        print(f"Val date: {valdate.strftime(dates.DATE_FORMAT)}")
        print("Vol surface information")
        surface_data.pretty_print()
        print(f"PDE time steps: {pde_config.n_timesteps}")
        print(f"PDE spot steps: {pde_config.n_meshes}")
        print(f"Optimizer: {method}")
        print(f"Tolerance: {tol}")

    # Loop over expiries
    pde_vols = []
    sol_as_init = config['sol_as_init']
    sol = None
    for exp_idx in range(len(expiry_grid)):
        if verbose:
            print(f"Optimizing at expiry: {exp_idx}/{len(expiry_grid)}")

        # Set expiry
        obj_builder.set_expiry(exp_idx, old_x, old_dx, old_p)

        # Initial point for optimization
        if exp_idx == 0:
            params_init = lv.params(0)
        else:
            params_init = (sol if sol_as_init else lv.params(exp_idx))

        # Constraints
        bounds = lv.section(exp_idx).constraints()

        # Optimize
        optimizer = create_optimizer(method, tol=tol, ftol=ftols[exp_idx])
        # optimizer = MultiOptimizer(methods = ['L-BFGS-B', 'SLSQP'], mtol=1e-2, ftol=ftols[exp_idx])
        result = optimizer.minimize(objective, x0=params_init, bounds=bounds)
        sol = result.x # Optimum parameters

        # Set local vol to optimum
        lv.update_params(exp_idx, sol)

        # Recalculate on optimum to get/set optimum density
        rmse = objective(sol)

        # Prepare next iteration
        old_x, old_dx, old_p = obj_builder.new_x, obj_builder.new_dx, obj_builder.new_p

        ## Optional for display and diagnostics ##
        if verbose:
            print(f"Result x: {sol}")
            print(f"RMSE(prices): {rmse:.4f}")
            if disp_opt:
                print(f"Result f: {result.fun}")
                print(f"Func evals: {result['nfev']}")
                for key in result.keys():
                    print(key + "\n", result[key])

        # Retrieve RMSE on vols
        if calc_pde_vols:
            pde_vols.append(obj_builder.calculate_vols())

    return {'lv': lv, 'iv_data': surface_data, 'pde_vols': pde_vols}


class LvObjectiveBuilder:
    def __init__(self, lv: TimeInterpolatedLocalVol, expiry_grid: list[float], fwds: list[float],
                 strike_surface: list[list[float]], cf_price_surface:list[list[float]],
                 pde_config: PdeConfig, option_type: str='straddle'):
        self.expiry_grid = expiry_grid
        self.option_type = string_to_optiontype(option_type)

        # Check consistency of time grids
        if len(lv.t_grid) <= 1:
            raise ValueError("LV time grid only has 1 point: it must contain at least 2")

        if len(lv.t_grid) != len(self.expiry_grid):
            raise ValueError("Inconsistent sizes between LV time grid and expiries")

        if not isequal(lv.t_grid[0], 0.0):
            raise ValueError("LV time grid does not start at 0")

        for i in range(len(self.expiry_grid) - 1):
            if not isequal(self.expiry_grid[i], lv.t_grid[i + 1]):
                raise ValueError("Inconsistent time values between LV time grid and expiries")

        # Global variables
        self.cf_price_surface = cf_price_surface
        self.strike_surface = strike_surface
        self.fwds = fwds
        self.lv = lv
        self.pde_config = pde_config
        self.start_time = 0.0

        # Slice variables
        self.exp_idx = 0
        self.step_grid = None
        self.old_p, self.old_x, self.old_dx = None, None, 0.0
        self.new_p, self.new_x, self.new_dx = None, None, 0.0
        self.fwd = 0.0
        self.strikes, self.cf_prices, self.pde_prices = None, None, None
        self.rmse = 0.0

    def objective(self, params: npt.ArrayLike) -> float:
        """ Objective function for LvSection calibration """
        # Update params first so they're available to check. Alternatively we could pass them
        # to check_params() and only set them if they're ok.
        self.lv.update_params(self.exp_idx, params)
        is_ok, penalty = self.lv.check_params(self.exp_idx)

        if is_ok:
            # Need to distinguish if it's the first expiry or not, because if it's the first
            # expiry, we need to initialize the density.
            # The reason this is needed as that the initialization of the density uses a
            # mollifier whose volatility is estimated from the Local Vol. So different LVs
            # lead to a different initial density.
            if self.exp_idx == 0: # Do the initialization step, otherwise used already stored
                self.old_x, self.old_dx, self.old_p = self.initialize()

            x, dx, p = fpde.density_step(self.old_p, self.old_x, self.old_dx,
                                         self.step_grid, self.lv, self.pde_config)
            self.new_x = x
            self.new_p = p
            self.new_dx = dx

            # Calculate the PDE options at the next expiry
            s = self.fwd * np.exp(x)
            pde_prices = []
            for k in self.strikes: # ToDo: can we do this vectorially over the strikes?
                match self.option_type:
                    case OptionType.CALL:
                        payoff = np.maximum(s - k, 0.0)
                    case OptionType.PUT:
                        payoff = np.maximum(k - s, 0.0)
                    case OptionType.STRADDLE:
                        payoff = np.abs(s - k)
                    case _:
                        raise ValueError(f"Unsupported option type: {self.option_type}")

                weighted_payoff = payoff * p
                pde_prices.append(np.trapezoid(weighted_payoff, x))

            # Calculate the objective function at the next expiry
            rmse = metrics.rmse(pde_prices, self.cf_prices)
            self.pde_prices = pde_prices
            self.rmse = rmse
            # print(self.rmse)
            return rmse
        else:
            # In principle we should return a penalty number. However, it is not clear at the moment if
            # that penalty should come from the model (where we know the parameters) or the
            # objective function (where we know the problem). It might need to come from both.
            # For now we are using a problem-specific penalty, i.e. the value if all the model prices
            # were 0, assuming that should be much bigger than at any reasonable solution.
            # ToDo: Claude recommends using constants.FLOAT_INFTY. But didn't we use it before and
            #       it led to some problems and that's why we're doing this now? To be tested again.
            # return constants.FLOAT_INFTY
            return self.cf_prices.sum()

    def set_expiry(self, exp_idx: int, old_x: npt.ArrayLike, old_dx: float, old_p: npt.ArrayLike) -> None:
        """ Set calibration targets and previous density for next calibration expiry """
        self.exp_idx = exp_idx
        ts = self.start_time if self.exp_idx == 0 else self.expiry_grid[self.exp_idx - 1]
        te = self.expiry_grid[self.exp_idx]
        self.step_grid = timegrids.build_timegrid(ts, te, self.pde_config)

        self.fwd = self.fwds[self.exp_idx]
        self.strikes = self.strike_surface[self.exp_idx]
        self.cf_prices = self.cf_price_surface[self.exp_idx]

        self.old_x = old_x
        self.old_dx = old_dx
        self.old_p = old_p

    def initialize(self) -> tuple[npt.ArrayLike, float, npt.ArrayLike]:
        """ Initialize calibrator to first expiry by calculating the initial density at start_time """
        self.start_time = fpde.FWD_PDE_START_TIME
        if self.expiry_grid[0] <= self.start_time:
            raise RuntimeError("First expiry too early to use analytical start in forward PDE")

        # First spot grid
        old_x, old_dx, old_spot_idx = fpde.build_spotgrid(self.expiry_grid[0], self.lv, self.pde_config)

        # First density
        lnvol = self.lv.ivol_guess(self.start_time)
        old_p = fpde.lognormal_density(old_x, self.start_time, lnvol)

        return old_x, old_dx, old_p

    def calculate_vols(self) -> npt.ArrayLike:
        """ Calculate Black implied vols for PDE prices """
        expiry = self.expiry_grid[self.exp_idx]
        pde_vols = []
        for k, p in zip(self.strikes, self.pde_prices, strict=True):
            if self.option_type == 0:
                pde_vols.append(black.implied_vol(expiry, k, True, self.fwd, p))
            elif self.option_type == 1:
                pde_vols.append(black.implied_vol(expiry, k, False, self.fwd, p))
            else:
                call = (p - k + self.fwd) / 2.0
                pde_vols.append(black.implied_vol(expiry, k, True, self.fwd, call))

        return pde_vols


if __name__ == "__main__":
    print("Hello")


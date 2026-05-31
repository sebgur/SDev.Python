import datetime as dt
import numpy as np
import numpy.typing as npt
import logging
from enum import Enum
from scipy.optimize import least_squares
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
from sdevpy.instruments.constants import string_to_optiontype, OptionType
log = logging.getLogger(__name__)


def calibrate_lv_bysections(valdate: dt.datetime, name: str, config: dict, **kwargs) -> dict:
    """ Calibrate InterpolatedParamLocalVol type to market data """
    # Arguments
    # verbose = kwargs.get('verbose', False)
    # disp_opt = kwargs.get('disp_opt', False)
    calc_pde_vols = kwargs.get('calc_pde_vols', False)
    force_restart = config.get('force_restart', False)
    model_name = config.get('model_name', 'VSVI')

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
                                                  option_type=option_type, voltol=1e-5)

    # Initial LV: either from scratch or from existing
    lv_t_grid = [0.0] # LV time grid
    lv_t_grid.extend(expiry_grid[:-1])
    lv = lvf.load_param_lv(valdate, name, t_grid=lv_t_grid, folder=config.get('lv_folder', None),
                           force_new=force_restart, model_name=model_name)
    lv.name, lv.valdate, lv.snapdate = name, valdate, valdate

    # Set forward PDE
    pde_config = fpde.PdeConfig(n_timesteps=config['pde_timesteps'], n_meshes=config['pde_spotsteps'],
                                scheme='rannacher', rescale_x=True, rescale_p=True,
                                shift_forward=False)

    # Penalty type
    penalty_str = config.get('penalty_type', 'infinity').lower()
    match penalty_str:
        case 'model':
            penalty_type = PenaltyType.MODEL
        case 'prices':
            penalty_type = PenaltyType.PRICES
        case 'infinity':
            penalty_type = PenaltyType.INFINITY
        case _:
            raise ValueError(f"Unsupported penalty type: {penalty_str}")

    # Set objective
    obj_builder = LvObjectiveBuilder(lv, expiry_grid, fwds, strike_surface, cf_price_surface, pde_config,
                                     penalty_type=penalty_type)#, verbose=verbose)
    objective = obj_builder.objective

    # Optimizer settings
    method = config.get('optimizer', 'COBYLA')
    tol = config.get('tol', None)
    maxiter = config.get('maxiter', 100)
    popsize = config.get('popsize', 5)
    use_least_squares = (method.lower() == 'leastsquares')

    # Initialize PDE
    # old_x, old_dx, old_p = obj_builder.initialize()
    old_x, old_dx, old_p = None, None, None

    # if verbose:
    log.info(f"Val date: {valdate.strftime(dates.DATE_FORMAT)}")
    log.info(f"Model: {model_name}")
    log.info(f"PDE time steps: {pde_config.n_timesteps}")
    log.info(f"PDE spot steps: {pde_config.n_meshes}")
    log.info(f"Optimizer: {method}")
    log.info("-"*50)

    # Loop over expiries
    pde_vols = []
    sol_as_init = config['sol_as_init']
    sol = None
    for exp_idx in range(len(expiry_grid)):
        # Set expiry
        obj_builder.set_expiry(exp_idx, old_x, old_dx, old_p)

        # Initial point for optimization
        if exp_idx == 0:
            params_init = lv.params(0)
        else:
            params_init = (sol if sol_as_init else lv.params(exp_idx))

        # Constraints
        bounds = lv.section(exp_idx).constraints()

        # if verbose:
        log.info(f"Optimizing at expiry: {exp_idx}/{len(expiry_grid)}")
        log.info(f"Initial params: {params_init}")
        log.info(f"Bounds: {bounds}")

        # Optimize: Least-Squares or regular optimization
        if use_least_squares:
            lb, ub = bounds.lb, bounds.ub
            params_init_ls = np.clip(params_init, lb, ub) # Clip to bounds
            result = least_squares(obj_builder.residuals, x0=params_init_ls, bounds=(lb, ub),
                                   method="trf", max_nfev=maxiter, xtol=tol, ftol=ftols[exp_idx])
        else:
            optimizer = create_optimizer(method, tol=tol, ftol=ftols[exp_idx], maxiter=maxiter,
                                         popsize=popsize, atol=ftols[exp_idx])

            result = optimizer.minimize(objective, x0=params_init, bounds=bounds)

        sol = result.x # Optimum parameters

        # Set local vol to optimum
        lv.update_params(exp_idx, sol)

        # Recalculate on optimum to get/set optimum density
        rmse = objective(sol)

        # Prepare next iteration
        old_x, old_dx, old_p = obj_builder.new_x, obj_builder.new_dx, obj_builder.new_p

        ## Optional for display and diagnostics ##
        # if verbose:
        log.info(f"Result x: {sol}")
        log.info(f"RMSE(prices): {rmse:.4f}")
        log.info(f"Number evals: {obj_builder.n_evals}")
        log.info("-"*50)
        # if disp_opt:
        # print(f"Result f: {result.fun}")
        # print(f"Func evals: {result['nfev']}")
        # for key in result.keys():
        #     print(key + "\n", result[key])

        # Retrieve RMSE on vols
        if calc_pde_vols:
            pde_vols.append(obj_builder.calculate_vols())

    return {'lv': lv, 'iv_data': surface_data, 'pde_vols': pde_vols, 'history': obj_builder.get_history()}


class PenaltyType(Enum):
    MODEL = 0
    PRICES = 1
    INFINITY = 2


class LvObjectiveBuilder:
    def __init__(self, lv: TimeInterpolatedLocalVol, expiry_grid: list[float], fwds: list[float],
                 strike_surface: list[list[float]], cf_price_surface:list[list[float]],
                 pde_config: PdeConfig, option_type: str='straddle',
                 penalty_type: PenaltyType=PenaltyType.INFINITY):#, verbose: bool=False):
        self.expiry_grid = expiry_grid
        self.option_type = string_to_optiontype(option_type)
        # self.verbose = verbose

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
        self.penalty_type = penalty_type

        # Slice variables
        self.exp_idx = 0
        self.step_grid = None
        self.old_p, self.old_x, self.old_dx = None, None, 0.0
        self.new_p, self.new_x, self.new_dx = None, None, 0.0
        self.fwd = 0.0
        self.strikes, self.cf_prices, self.pde_prices = None, None, None
        self.rmse = 0.0
        self.n_evals = 0 # Keep track of number of objective evaluations
        self.history = [None] * len(self.expiry_grid)

    def calculate_pde_prices(self) -> npt.ArrayLike:
        """ Evolve the PDE and return the price vector """
        # Initialization at first expiry
        # The initialization needs to be optimized on because the density uses a
        # mollifier whose volatility is estimated from the Local Vol on which we optimize.
        if self.exp_idx == 0: # Do the initialization step, otherwise used already stored
            self.old_x, self.old_dx, self.old_p = self.initialize()

        x, dx, p = fpde.density_step(self.old_p, self.old_x, self.old_dx,
                                    self.step_grid, self.lv, self.pde_config)
        self.new_x, self.new_p, self.new_dx = x, p, dx

        # Calculate the PDE options at the next expiry
        pde_prices = fpde.vanilla_expectation(self.fwd, p, x, self.strikes, self.option_type)
        self.pde_prices = pde_prices
        self.rmse = metrics.rmse(pde_prices, self.cf_prices) # For logging
        self.history[self.exp_idx]['evals'].append(self.n_evals)
        self.history[self.exp_idx]['rmses'].append(self.rmse)
        # if self.verbose:
        log.debug(f"RMSE: {self.rmse}")

    def residuals(self, params: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """ Residual vector for least_squares: pde_price - cf_price per strike """
        self.n_evals += 1

        # Update parameters
        self.lv.update_params(self.exp_idx, params)
        is_ok, penalty = self.lv.check_params(self.exp_idx)

        # if self.verbose:
        log.debug(f"> Trial{self.n_evals}: {np.array2string(np.asarray(params), precision=8)}")

        if is_ok:
            self.calculate_pde_prices()
            return self.pde_prices - np.asarray(self.cf_prices)
        else:
            # if self.verbose:
            log.debug(" Rejected")
            return np.full_like(self.cf_prices, np.sqrt(self.cf_prices.sum()))

    def objective(self, params: npt.ArrayLike) -> float:
        """ Objective function for LvSection calibration """
        self.n_evals += 1

        # Update parameters
        self.lv.update_params(self.exp_idx, params)
        is_ok, mod_penalty = self.lv.check_params(self.exp_idx)

        # if self.verbose:
        log.debug(f"> Trial{self.n_evals}: {np.array2string(np.asarray(params), precision=8)}")

        if is_ok:
            self.calculate_pde_prices()
            return self.rmse
        else:
            # The penalty approach is left to the user's choice. It can:
            #   * come from the model (where we know the parameters), i.e. mod_penalty
            #   * come from objective function (where we know the problem)
            #   * be set to infinity.
            match self.penalty_type:
                case PenaltyType.MODEL:
                    eff_penalty = mod_penalty
                case PenaltyType.PRICES:
                    eff_penalty = self.cf_prices.sum()
                case PenaltyType.INFINITY:
                    eff_penalty = constants.FLOAT_INFTY
                case _:
                    raise ValueError(f"Unsupported penalty type: {self.penalty_type}")

            # if self.verbose:
            log.debug(f" Rejected, penalty = {eff_penalty}")
            return eff_penalty

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

        self.n_evals = 0
        self.history[self.exp_idx] = {'evals': [], 'rmses': []}

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

    # def reset_history(self) -> None:
    #     self.history = [None] * len(self.expiry_grid)

    def get_history(self) -> list[dict]:
        return self.history

    def calculate_vols(self) -> npt.ArrayLike:
        """ Calculate Black implied vols for PDE prices """
        expiry = self.expiry_grid[self.exp_idx]
        pde_vols = []
        for k, p in zip(self.strikes, self.pde_prices, strict=True):
            match self.option_type:
                case OptionType.CALL:
                    pde_vols.append(black.implied_vol(expiry, k, True, self.fwd, p))
                case OptionType.PUT:
                    pde_vols.append(black.implied_vol(expiry, k, False, self.fwd, p))
                case OptionType.STRADDLE:
                    call = (p - k + self.fwd) / 2.0
                    pde_vols.append(black.implied_vol(expiry, k, True, self.fwd, call))
                case _:
                    raise ValueError(f"Unknown option type: {self.option_type}")
            # if self.option_type == 0:
            #     pde_vols.append(black.implied_vol(expiry, k, True, self.fwd, p))
            # elif self.option_type == 1:
            #     pde_vols.append(black.implied_vol(expiry, k, False, self.fwd, p))
            # else:
            #     call = (p - k + self.fwd) / 2.0
            #     pde_vols.append(black.implied_vol(expiry, k, True, self.fwd, call))

        return pde_vols


if __name__ == "__main__":
    print("Hello")


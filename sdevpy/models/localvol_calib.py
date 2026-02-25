import os
import datetime as dt
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sdevpy.tools import timegrids, dates
from sdevpy.models import svivol, biexp
from sdevpy.models import localvol
from sdevpy.models import localvol_factory as lvf
from sdevpy.pde import forwardpde as fpde
from sdevpy.analytics import black
from sdevpy.maths import metrics
from sdevpy.maths.optimization import *
from sdevpy.market import volsurface as vsurf


########## ToDo ########################################################################
# * Implement MC and check calibration against it.
# * Test with calibrated XYZ in CubicVol if we can do better manually
# * Add 1d solving to ATM only, to do live and Vega with smile solving less often.
# * Use actual data from SPX
# * Calibration weights based on percentiles, with possible removal of options
# * Use seaborn to represent diffs between IV and LV prices on quoted pillars
# * Upload to pypi, make Colab, post.

IS_CALL = True

def calibrate_lv(valdate, name, config, **kwargs):
    # Arguments
    verbose = kwargs.get('verbose', False)
    disp_opt = kwargs.get('disp_opt', False)
    calc_pde_vols = kwargs.get('calc_pde_vols', False)

    # Retrieve target market option data
    file = vsurf.data_file(vsurf.test_data_folder(), name, valdate)
    surface_data = vsurf.vol_surface(file)
    expiries = surface_data.expiries
    fwds = surface_data.forwards
    strike_surface = surface_data.get_strikes('absolute')
    vol_surface = surface_data.vols

    # Set calibration time grid
    expiry_grid = np.array([timegrids.model_time(valdate, expiry) for expiry in expiries])

    # Set calibration targets
    cf_price_surface, ftols = calibration_targets(expiry_grid, fwds, strike_surface, vol_surface)

    # Initial LV: either from scratch or from existing
    if config['start_new']:
        lv = lvf.load_lv_new(expiry_grid, config['model'])
    else:
        lv = lvf.load_lv_from_folder(expiry_grid, valdate, name, config['lv_folder'])
    lv.name, lv.valdate, lv.snapdate = name, valdate, valdate

    # Set forward PDE
    mesh_vol = vol_surface.mean()
    pde_config = fpde.PdeConfig(n_time_steps=config['pde_timesteps'], n_meshes=config['pde_spotsteps'],
                                mesh_vol=mesh_vol, scheme='rannacher', rescale_x=True, rescale_p=True)

    # Set objective
    obj_builder = LvObjectiveBuilder(lv, fwds, strike_surface, cf_price_surface, pde_config)
    objective = obj_builder.objective

    # Optimizer settings
    method = config['optimizer']
    tol = config['tol']

    # Initialize PDE
    old_x, old_dx, old_p = obj_builder.initialize()

    if verbose:
        print(f"Val date: {valdate.strftime(dates.DATE_FORMAT)}")
        print(f"Vol surface information")
        surface_data.pretty_print()
        print(f"Mesh vol: {mesh_vol*100:.2f}%")
        print(f"PDE time steps: {pde_config.n_time_steps}")
        print(f"PDE spot steps: {pde_config.n_meshes}")
        print(f"Optimizer: {method}")
        print(f"Tolerance: {tol}")

    # Loop over expiries
    pde_vols = []
    sol_as_init = config['sol_as_init']
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
                    if key in result:
                        print(key + "\n", result[key])

        # Retrieve RMSE on vols
        if calc_pde_vols:
            pde_vols.append(obj_builder.calculate_vols())

    return {'lv': lv, 'iv_data': surface_data, 'pde_vols': pde_vols}


class LvObjectiveBuilder:
    def __init__(self, lv, fwds, strike_surface, cf_price_surface, pde_config):
        self.expiry_grid = lv.t_grid
        self.cf_price_surface = cf_price_surface
        self.strike_surface = strike_surface
        self.fwds = fwds
        self.lv = lv
        self.pde_config = pde_config
        self.start_time = 0.0

        # Slice variable
        self.exp_idx = 0
        self.step_grid = None
        self.old_p = None
        self.old_x = None
        self.old_dx = 0.0
        self.new_p = None
        self.new_x = None
        self.new_dx = 0.0
        self.fwd = 0.0
        self.strikes = None
        self.cf_prices = None
        self.pde_prices = None
        self.rmse = 0.0

    def objective(self, params):
        # Update params first so they're available to check. Alternatively we could pass them
        # to check_params() and only set them if they're ok.
        self.lv.update_params(self.exp_idx, params)
        is_ok, penalty = self.lv.check_params(self.exp_idx)

        if is_ok:
            x, dx, p = fpde.density_step(self.old_p, self.old_x, self.old_dx,
                                         self.step_grid, self.lv.value, self.pde_config)
            self.new_x = x
            self.new_p = p
            self.new_dx = dx

            # Calculate the PDE options at the next expiry
            s = self.fwd * np.exp(x)
            pde_prices = []
            for k in self.strikes: # ToDo: can we do this vectorially over the strikes?
                payoff = np.maximum(s - k, 0.0) # ToDo: accept calls and puts
                weighted_payoff = payoff * p
                pde_prices.append(np.trapezoid(weighted_payoff, x))

            # Calculate the objective function at the next expiry
            rmse = metrics.rmse(pde_prices, self.cf_prices)
            self.pde_prices = pde_prices
            self.rmse = rmse
            return rmse
        else:
            # In principle we should return a penalty number. However, it is not clear at the moment if
            # that penalty should come from the model (where we know the parameters) or the
            # objective function (where we know the problem). It might need to come from both.
            # For now we are using a problem-specific penalty, i.e. the value if all the model prices
            # were 0, assuming that should be much bigger than at any reasonable solution.
            return self.cf_prices.sum()

    def set_expiry(self, exp_idx, old_x, old_dx, old_p):
        self.exp_idx = exp_idx
        ts = self.start_time if exp_idx == 0 else self.expiry_grid[exp_idx - 1]
        te = self.expiry_grid[exp_idx]
        self.step_grid = timegrids.build_timegrid(ts, te, self.pde_config)

        self.fwd = self.fwds[exp_idx]
        self.strikes = self.strike_surface[exp_idx]
        self.cf_prices = self.cf_price_surface[exp_idx]

        self.old_x = old_x
        self.old_dx = old_dx
        self.old_p = old_p

    def initialize(self, start_time=1.0/365.0):
        self.start_time = start_time
        if self.expiry_grid[0] <= start_time:
            raise RuntimeError("First expiry too early to use analytical start in forward PDE")

        # First spot grid
        old_x, old_dx, old_spot_idx = fpde.build_spotgrid(self.expiry_grid[0], self.pde_config)

        # First density
        old_p = fpde.lognormal_density(old_x, self.start_time, self.pde_config.mesh_vol)

        return old_x, old_dx, old_p

    def calculate_vols(self):
        expiry = self.expiry_grid[self.exp_idx]
        pde_vols = []
        for k, p in zip(self.strikes, self.pde_prices):
            pde_vols.append(black.implied_vol(expiry, k, IS_CALL, self.fwd, p))

        return pde_vols


def calibration_targets(expiry_grid, fwds, strike_surface, vol_surface):
    cf_price_surface = []
    ftols = []
    itol = 1e-6 # 1bp
    for exp_idx, expiry in enumerate(expiry_grid):
        fwd = fwds[exp_idx]
        strikes = strike_surface[exp_idx]
        vols = vol_surface[exp_idx]
        cf_price = black.price(expiry, strikes, IS_CALL, fwd, vols)
        cf_price_surface.append(cf_price)
        vols = vols + itol
        cf_price_bump = black.price(expiry, strikes, IS_CALL, fwd, vols)
        ftols.append(metrics.rmse(cf_price, cf_price_bump))

    return cf_price_surface, ftols


if __name__ == "__main__":
    verbose, n_digits = False, 6
    np.set_printoptions(suppress=True, precision=n_digits)
    name = "XYZ"
    valdate = dt.datetime(2025, 12, 15)
    lv_data_folder = lvf.test_data_folder()
    # 'L-BFGS-B'
    config = {'start_new': False, 'model': 'CubicVol', 'store_date': valdate,
              'optimizer': 'SLSQP', 'tol': 1e-4, 'pde_timesteps': 50,
              'pde_spotsteps': 100, 'lv_folder': lv_data_folder,
              'sol_as_init': False}

    # Calibrate LV
    calib_result = calibrate_lv(valdate, name, config, verbose=True, calc_pde_vols=True)
    lv = calib_result['lv']

    # Dump LV result to file
    out_folder = lvf.test_data_folder()
    fname = valdate.strftime(dates.DATE_FILE_FORMAT) + "." + name + "." + config['model']
    out_file = os.path.join(out_folder, fname + ".json")
    lv.dump(out_file)

    # ################ DIAGNOSTICS ################################################################
    # Retrieve results for diagnostics
    pde_vols = calib_result['pde_vols']
    surface_data = calib_result['iv_data']
    expiries = surface_data.expiries
    expiry_grid = np.array([timegrids.model_time(valdate, expiry) for expiry in expiries])
    # fwds = surface_data.forwards
    strike_surface = surface_data.get_strikes('absolute')
    vol_surface = surface_data.vols

    # Calculate RMSEs on vols
    vol_rmses = []
    for exp_idx in range(len(expiry_grid)):
        vol_rmses.append(10000.0 * metrics.rmse(vol_surface[exp_idx], pde_vols[exp_idx]))

    # Display price results
    n_rows, n_cols = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            exp_idx = n_cols * i + j
            strikes = strike_surface[exp_idx]
            ax.plot(strikes, pde_vols[exp_idx], label="PDE", color='red')
            ax.plot(strikes, vol_surface[exp_idx], label="CF", color='blue')
            ax.set_title(f"T:{expiry_grid[exp_idx]:.2f}, RMSE: {vol_rmses[exp_idx]:.4f}")
            ax.set_xlabel('strike')
            ax.set_ylabel('price')
            ax.legend()

    fig.suptitle('Option prices, PDE vs CF', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Display LV results
    n_rows, n_cols = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            exp_idx = n_cols * i + j
            expiry = expiry_grid[exp_idx]
            vol = vol_surface[exp_idx].mean()
            stdev = vol * np.sqrt(expiry)
            print(f"Params at {expiry:.3f}, {lv.params(exp_idx)}")
            xs = np.linspace(-3.0 * stdev, 3.0 * stdev, 100)
            lvs = lv.value(expiry, xs)
            ax.plot(xs, lvs, label="LV", color='blue')
            # strikes = strike_surface[exp_idx]
            # lvs = lv.value(expiry_grid[exp_idx], np.log(strikes / fwds[exp_idx]))
            # ax.plot(strikes, lvs, label="LV", color='blue')
            ax.set_title(f"T:{expiry:.2f}, RMSE: {vol_rmses[exp_idx]:.4f}")
            ax.set_xlabel('strike')
            ax.set_ylabel('price')
            ax.legend()

    fig.suptitle('Option prices, PDE vs CF', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

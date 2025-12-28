import datetime as dt
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sdevpy.models import svivol
from sdevpy.tools import timegrids
from sdevpy.models import localvol
from sdevpy.pde import forwardpde as fpde
from sdevpy.analytics import black
from sdevpy.maths import metrics
from sdevpy.maths.optimization import *


########## ToDo (calibration) #################################################
# * Isn't Rannacher's scheme defined the other way around for forward PDE?
#   Make sure Rannacher throws an error if its time def is not given.
# * Refresh optimizer implementation, get definition/control of stopping criteria.
# * Implement calibration by sections
# * Use seaborn to represent diffs between IV and LV prices on quoted pillars
# * Add 1d solving to ATM only, to do live and Vega with smile solving less often.
# * During the warmup in the time direction, we can allocate
#   on each time slice a local vol functional form that is only a function
#   of the spot. This would be a generalized version of the storage
#   of the time interpolation indices for an interpolated surface.
# * Resolve forward by taking the previous parametric form as starting point.
# * To check the quality of the calibration, start by comparing against same forward
#   PDE as used in calibration. Define a simple method that calculates the whole
#   surface. Then implement and check against backward PDE.
# * Make notebook that illustrates the whole flow.
# * Introduce unit testing. Cleanup package, upload to pypi.
# * Implement input option filters (time and percentiles)
# * Make Colab, post.

def generate_sample_data(valdate, terms):
    spot, r, q = 100.0, 0.04, 0.02
    percents = [0.10, 0.25, 0.5, 0.75, 0.90]
    base_vol = 0.25
    expiries, fwds, strike_surface, vol_surface = [], [], [], []
    for term in terms:
        expiry = valdate + dt.timedelta(days=int(term * 365.25))
        fwd = spot * np.exp((r - q) * term)
        base_std = base_vol * np.sqrt(term)
        a, b, rho, m, sigma = base_vol, 0.1, 0.0, 0.5, 0.25
        strikes, vols = [], []
        for p in percents:
            logm = -0.5 * base_std**2 + base_std * norm.ppf(p)
            strikes.append(fwd * np.exp(logm))
            vols.append(svivol.svivol(term, logm, a, b, rho, m, sigma))

        expiries.append(expiry)
        fwds.append(fwd)
        strike_surface.append(strikes)
        vol_surface.append(vols)

    return np.array(expiries), np.array(fwds), np.array(strike_surface), np.array(vol_surface)


class LvObjectiveBuilder:
    def __init__(self, lv, expiry_grid, fwds, strike_surface, cf_price_surface, pde_config):
        self.expiry_grid = expiry_grid
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
        self.lv.update_params(self.exp_idx, params)

        x, dx, p = fpde.density_step(self.old_p, self.old_x, self.old_dx,
                                     self.step_grid, self.lv.value, self.pde_config)
        self.new_x = x
        self.new_p = p
        self.new_dx = dx
        # new_lv = lv.value(te, x)

        # Calculate the PDE options at the next expiry
        s = self.fwd * np.exp(x)
        pde_prices = []
        for k in self.strikes: # ToDo: can we do this vectorially over the strikes?
            payoff = np.maximum(s - k, 0.0)
            weighted_payoff = payoff * p
            pde_prices.append(np.trapezoid(weighted_payoff, x))

        # Calculate the objective function at the next expiry
        rmse = metrics.rmse(pde_prices, self.cf_prices)
        self.pde_prices = pde_prices
        self.rmse = rmse
        return rmse

    def set_expiry(self, exp_idx, old_x, old_dx, old_p):
        self.exp_idx = exp_idx
        # expiry = expiry_grid[exp_idx]
        ts = self.start_time if exp_idx == 0 else self.expiry_grid[exp_idx - 1]
        te = self.expiry_grid[exp_idx]
        self.step_grid = fpde.build_timegrid(ts, te, self.pde_config)

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
        # self.set_expiry(0, old_x, old_dx, old_p)



if __name__ == "__main__":
    ##### Create IV and forward target data #############################################
    valdate = dt.datetime(2025, 12, 15)
    terms = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    expiries, fwds, strike_surface, vol_surface = generate_sample_data(valdate, terms)

    print(f"Val date: {valdate.strftime("%Y-%b-%d")}")
    print(f"Expiries: {expiries.shape}")
    # print(f"Expiries: {[d.strftime("%Y-%b-%d") for d in expiries]}")
    print(f"Forwards: {fwds.shape}")
    print("Strikes", strike_surface.shape)
    print(f"Vols: {vol_surface.shape}")

    ##### Calibration to target data ####################################################
    # Calibration time grid
    expiry_grid = np.array([timegrids.model_time(valdate, expiry) for expiry in expiries])

    # Calibration targets
    cf_price_surface = []
    is_call = True
    for exp_idx, expiry in enumerate(expiry_grid):
        fwd = fwds[exp_idx]
        strikes = strike_surface[exp_idx]
        vols = vol_surface[exp_idx]
        cf_price = black.price(expiry, strikes, is_call, fwd, vols)
        cf_price_surface.append(cf_price)

    # Create an LV with suitable slices
    section_grid = [svivol.SviVolSection() for i in range(len(expiry_grid))]
    lv = localvol.InterpolatedParamLocalVol(expiry_grid, section_grid)

    ## Set up forward PDE ##
    mesh_vol = vol_surface.mean()
    print(f"Mesh vol: {mesh_vol*100:.2f}%")
    pde_config = fpde.PdeConfig(n_time_steps=50, n_meshes=250, mesh_vol=mesh_vol, scheme='rannacher',
                                rescale_x=True, rescale_p=True)
    print(f"Time steps: {pde_config.n_time_steps}")
    print(f"Spot steps: {pde_config.n_meshes}")

    # Objective builder
    obj_builder = LvObjectiveBuilder(lv, expiry_grid, fwds, strike_surface,
                                     cf_price_surface, pde_config)

    # Get objective
    objective = obj_builder.objective

    # Optimizer config
    lw_bounds = [0.0, 0.0, -0.99, -1.0, 0.0] # a, b, rho, m, sigma
    up_bounds = [0.8, 1.0, 0.99, 1.0, 1.0] # a, b, rho, m, sigma
    # method = 'Nelder-Mead'
    # method = 'Powell'
    method = 'L-BFGS-B'
    tol = 1e-4
    optimizer = create_optimizer(method, tol=tol)
    bounds = opt.Bounds(lw_bounds, up_bounds, keep_feasible=False)

    # Initialize PDE
    old_x, old_dx, old_p = obj_builder.initialize()

    # Initial parameters for the first expiry
    params_init = svivol.sample_params(expiry_grid[0])

    # Loop over expiries
    for exp_idx in range(len(expiry_grid)):
        print(f"Expiry: {exp_idx}")
        obj_builder.set_expiry(exp_idx, old_x, old_dx, old_p)

        # Calculate objective
        # old_x = obj_builder.old_x
        # old_p = obj_builder.old_p
        # rmse = objective(params_init)
        # new_x = obj_builder.new_x
        # new_p = obj_builder.new_p
        # print(rmse)

        # plt.plot(old_x, old_p, color='blue', label='old')
        # plt.plot(new_x, new_p, color='red', label='new')
        # plt.legend(loc='upper right')
        # plt.show()

        # Optimize
        optimizer = create_optimizer(method, tol=tol)
        result = optimizer.minimize(objective, x0=params_init, bounds=bounds)
        sol = result.x
        fun = result.fun
        # print(f"Result x: {sol}")
        print(f"Result f: {fun}")

        # Set local vol to optimum
        lv.update_params(exp_idx, sol)

        # Recalculate on optimum to get optimum density
        rmse = objective(sol)
        print(rmse)

        # Prepare next iteration
        params_init = sol # Use the solution as initial point for next iteration
        old_x = obj_builder.new_x
        old_dx = obj_builder.new_dx
        old_p = obj_builder.new_p

        # Check optimization result
        # y = objective(x)
        # print(f"Result f check: {y}")
        # print(f"Result f check: {obj_builder.rmse}")
        cf_p = np.asarray(obj_builder.cf_prices)
        pde_p = np.asarray(obj_builder.pde_prices)
        print(cf_p)
        print(pde_p)

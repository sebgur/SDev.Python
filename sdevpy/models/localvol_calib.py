import datetime as dt
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sdevpy.models import svivol, biexp
from sdevpy.tools import timegrids
from sdevpy.models import localvol
from sdevpy.pde import forwardpde as fpde
from sdevpy.analytics import black
from sdevpy.maths import metrics
from sdevpy.maths.optimization import *


########## ToDo (calibration) #################################################
# * Isn't Rannacher's scheme defined the other way around for forward PDE?
#   Make sure Rannacher throws an error if its time def is not given.
# * Get definition/control of optimizer's stopping criteria.
# * Use seaborn to represent diffs between IV and LV prices on quoted pillars
# * Add 1d solving to ATM only, to do live and Vega with smile solving less often.
# * To check the quality of the calibration, start by comparing against same forward
#   PDE as used in calibration. Define a simple method that calculates the whole
#   surface. Then implement and check against backward PDE.
# * Implement input option filters (time and percentiles)
# * Introduce unit testing. Cleanup package, upload to pypi.
# * Make Colab, post.


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
        is_ok, penalty = self.lv.check_params(self.exp_idx)

        if is_ok:
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
        else:
            return penalty

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
    expiries, fwds, strike_surface, vol_surface = svivol.generate_sample_data(valdate, terms)

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
    section_grid = [biexp.BiExpSection() for i in range(len(expiry_grid))]
    # section_grid = [svivol.SviVolSection() for i in range(len(expiry_grid))]
    lv = localvol.InterpolatedParamLocalVol(expiry_grid, section_grid)

    ## Set up forward PDE ##
    mesh_vol = vol_surface.mean()
    print(f"Mesh vol: {mesh_vol*100:.2f}%")
    # Original trial: n_times = 50, n_meshes = 250
    pde_config = fpde.PdeConfig(n_time_steps=50, n_meshes=100, mesh_vol=mesh_vol, scheme='rannacher',
                                rescale_x=True, rescale_p=True)
    print(f"Time steps: {pde_config.n_time_steps}")
    print(f"Spot steps: {pde_config.n_meshes}")

    # Objective builder
    obj_builder = LvObjectiveBuilder(lv, expiry_grid, fwds, strike_surface,
                                     cf_price_surface, pde_config)

    # Get objective
    objective = obj_builder.objective

    # Constraints
    # lw_bounds = [0.0, 0.0, -0.99, -1.0, 0.0] # a, b, rho, m, sigma
    # up_bounds = [0.8, 5.0, 0.99, 2.0, 1.0] # a, b, rho, m, sigma
    lw_bounds = [0.01, 0.01, 0.01, 0.01, 0.01, -2.0] # a, b, rho, m, sigma
    up_bounds = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0] # a, b, rho, m, sigma

    # Optimizer
    # method = 'Nelder-Mead'
    # method = 'Powell'
    # method = 'DE'
    method = 'SLSQP'
    # method = 'L-BFGS-B'
    tol = 1e-8
    atol = 1e-2
    optimizer = create_optimizer(method, tol=tol, atol=atol)
    bounds = opt.Bounds(lw_bounds, up_bounds, keep_feasible=False)

    # Initialize PDE
    old_x, old_dx, old_p = obj_builder.initialize()

    # Initial parameters for the first expiry
    # params_init = svivol.sample_params(expiry_grid[0], mesh_vol)
    params_init = biexp.sample_params(expiry_grid[0], mesh_vol)
    params_init = [params_init] * len(expiry_grid)
    # print(params_init)

    # params_init = [[0.237, 3.3, -0.22, -0.007, 0.0],
    #                [0.001, 3.27, 0.05, 0.0125, 0.11],
    #                [0.089, 0.755,  0.19, 0.07,  0.23],
    #                [0.18605, 0.26973, 0.45985, 0.22791, 0.3253],
    #                [0.24804566, 0.06740734, 0.60760707, 0.4204044,  0.37236663],
    #                [0.27136465, 0.0025457,  0.87826455, 1., 0.58212948]]
    # # params_init = [0.179, 3.3, -0.40, -0.019, 0.025]
    # # print(params_init)
    # # obj_builder.set_expiry(0, old_x, old_dx, old_p)
    # # rmse = objective(params_init)
    # # print(rmse)

    # Loop over expiries
    rmses = []
    cf_prices, cf_vols = [], []
    pde_prices, pde_vols = [], []
    params_init = params_init[0]
    for exp_idx in range(len(expiry_grid)):
        print(f"Optimizing at expiry: {exp_idx}/{len(expiry_grid)}")
        # Set expiry
        obj_builder.set_expiry(exp_idx, old_x, old_dx, old_p)

        # Optimize
        optimizer = create_optimizer(method, tol=tol)
        result = optimizer.minimize(objective, x0=params_init, bounds=bounds)
        sol = result.x # Optimum parameters
        # fun = result.fun
        print(f"Result x: {sol}")
        # print(f"Result f: {fun}")

        # Set local vol to optimum
        lv.update_params(exp_idx, sol)

        # Recalculate on optimum to get optimum density
        rmse = objective(sol)
        # rmses.append(rmse)
        print(f"RMSE at exp idx {exp_idx}: {rmse:.4f}")

        # Prepare next iteration
        params_init = sol # Use the solution as initial point for next iteration
        old_x = obj_builder.new_x
        old_dx = obj_builder.new_dx
        old_p = obj_builder.new_p

        # Check optimization result
        # y = objective(x)
        # print(f"Result f check: {y}")
        # print(f"Result f check: {obj_builder.rmse}")
        cf_prices.append(obj_builder.cf_prices)
        pde_prices_at_exp = obj_builder.pde_prices
        pde_prices.append(pde_prices_at_exp)
        cf_vols.append(vol_surface[exp_idx])
        # pde_vols.append(vol_surface[exp_idx]) # ToDo: transform prices
        pde_vols_at_exp = []
        expiry = expiry_grid[exp_idx]
        fwd = fwds[exp_idx]
        strikes = strike_surface[exp_idx]
        for k, p in zip(strikes, pde_prices_at_exp):
            pde_vols_at_exp.append(black.implied_vol(expiry, k, is_call, fwd, p))

        pde_vols.append(pde_vols_at_exp)
        rmses.append(10000.0 * metrics.rmse(cf_vols, pde_vols))

    # Display price results
    n_rows, n_cols = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            exp_idx = n_cols * i + j
            strikes = strike_surface[exp_idx]
            ax.plot(strikes, pde_vols[exp_idx], label="PDE", color='red')
            ax.plot(strikes, cf_vols[exp_idx], label="CF", color='blue')
            ax.set_title(f"T:{expiry_grid[exp_idx]:.2f}, RMSE: {rmses[exp_idx]:.4f}")
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
            print(f"Params at {expiry:.3f}: {lv.params(exp_idx)}")
            xs = np.linspace(-3.0 * stdev, 3.0 * stdev, 100)
            lvs = lv.value(expiry, xs)
            ax.plot(xs, lvs, label="LV", color='blue')
            # strikes = strike_surface[exp_idx]
            # lvs = lv.value(expiry_grid[exp_idx], np.log(strikes / fwds[exp_idx]))
            # ax.plot(strikes, lvs, label="LV", color='blue')
            ax.set_title(f"T:{expiry:.2f}, RMSE: {rmses[exp_idx]:.4f}")
            ax.set_xlabel('strike')
            ax.set_ylabel('price')
            ax.legend()

    fig.suptitle('Option prices, PDE vs CF', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

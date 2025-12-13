import datetime as dt
import numpy as np
from scipy.stats import norm
from sdevpy.models import svi
from sdevpy.tools import timegrids
from sdevpy.models import localvol
from sdevpy.pde import forwardpde as fpde

########## ToDo (calibration) #################################################
# * Create objective function, constraints, with pre-calculation of weighted
#   payoff that doesn't depend on vols.
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
        a, b, rho, m, sigma = base_std**2, 0.1, 0.0, 0.5, 0.25
        strikes, vols = [], []
        for p in percents:
            logm = -0.5 * base_std**2 + base_std * norm.ppf(p)
            strikes.append(fwd * np.exp(logm))
            vols.append(svi.svi(term, logm, a, b, rho, m, sigma))

        expiries.append(expiry)
        fwds.append(fwd)
        strike_surface.append(strikes)
        vol_surface.append(vols)

    return np.array(expiries), np.array(fwds), np.array(strike_surface), np.array(vol_surface)


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

    # Create an LV with suitable slices
    section_grid = [svi.SviSection() for i in range(len(expiry_grid))]
    lv = localvol.InterpolatedParamLocalVol(expiry_grid, section_grid)

    ## Set up forward PDE ##
    mesh_vol = vol_surface.mean()
    print(f"Mesh vol: {mesh_vol*100:.2f}%")
    pde_config = fpde.PdeConfig(n_time_steps=50, n_meshes=250, mesh_vol=mesh_vol, scheme='rannacher',
                                rescale_x=True, rescale_p=True)
    print(f"Time steps: {pde_config.n_time_steps}")
    print(f"Spot steps: {pde_config.n_meshes}")

    # Spot grid
    x, dx, spot_idx = fpde.build_spotgrid(expiry_grid[0], pde_config)

    # Time grid

    ## Bootstrap initialization ##
    # Initiate a probability density at t = 0 to start the forward PDE
    start_time = 1.0 / 365.0
    if expiry_grid[0] <= start_time:
        raise RuntimeError("First expiry too early to use analytical start in forward PDE")

    p = fpde.lognormal_density(x, start_time, pde_config.mesh_vol)

    # Initial parameters for the first expiry

    # Initialize the LV slice
    # Use it to calculate the probability density at the next expiry
    # Calculate the PDE options at the next expiry
    # Calculate the objective function at the next expiry
    # Optimize the objective function
    # Retrieve optimum parameters and optimum density
    # Iterate: use previous initial parameters as starting points
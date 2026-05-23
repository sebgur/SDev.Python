import datetime as dt
import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from sdevpy.pde import pdeschemes
from sdevpy.pde.pdeschemes import PdeConfig
from sdevpy.utilities import timegrids
from sdevpy.utilities.tools import isequal
from sdevpy.market.eqforward import EqForwardCurve
from sdevpy.volatility.impliedvol.impliedvol import ImpliedVol
from sdevpy.volatility.localvol.localvol import LocalVol
from sdevpy.instruments.constants import OptionType, string_to_optiontype


def density_step(old_p: npt.NDArray[np.float64], old_x: npt.NDArray[np.float64], old_dx: float,
                 t_grid: npt.NDArray[np.float64], local_vol, config: PdeConfig):
    """ Forward PDE evolution along t_grid, with optional rescaling of meshes at
        beginning of the step, optional shifting of the density along its x-axis
        to mach the forward, and optional rescaling of density to integrated to 1
        at end of the step """
    # Rescale spot grid
    if config.rescale_x:
        x, dx, spot_idx = build_spotgrid(t_grid[-1], config)
        p = np.interp(x, old_x, old_p, left=0.0, right=0.0)
    else:
        x, dx, p = old_x, old_dx, old_p

    # Forward reduction
    for i in range(t_grid.shape[0] - 1):
        p = roll_forward(p, x, dx, t_grid[i], t_grid[i + 1], local_vol, config)

    # Shift to match forward
    if config.shift_forward:
        p = shift_forward(x, p)

    # Rescale density
    if config.rescale_p: # Rescale mass to 1.0 at te
        #mass = mass(p, x) # np.trapezoid(p, x)
        p /=  mass(p, x)

    return x, dx, p


def density(maturity: float, local_vol, config: PdeConfig):
    """ Simple forward PDE for density, without any rescaling """
    # Build time grid
    t_grid = timegrids.build_timegrid(0.0, maturity, config)
    n_timegrid = t_grid.shape[0]

    # Build spot grid
    x, dx, spot_idx = build_spotgrid(maturity, config)

    # Initial probability
    p = lognormal_density(x, 1.0 / 365.0, config.mesh_vol)

    # Forward reduction
    for i in range(n_timegrid - 1):
        p = roll_forward(p, x, dx, t_grid[i], t_grid[i + 1], local_vol, config)

    return x, dx, p


def build_spotgrid(maturity: float, config: PdeConfig) -> tuple[npt.ArrayLike, float, int]:
    """ Build spot grid for PDEs """
    mesh_percentile = config.percentile
    if config.iv_surface is None:
        mesh_vol = config.mesh_vol
    else:
        mesh_vol = config.iv_surface.black_volatility(maturity, 1.0, 1.0)

    n_meshes = config.n_meshes
    p = norm.ppf(1.0 - mesh_percentile)
    x_max = mesh_vol * np.sqrt(maturity) * p
    n_half = int(n_meshes / 2)
    dx = x_max / n_half
    x_grid = np.zeros(2 * n_half + 1)
    x = 0.0
    for i in range(n_half):
        x = x + dx
        x_grid[n_half + 1 + i] = x
        x_grid[n_half - 1 - i] = -x

    return x_grid, dx, n_half


def lognormal_density(x: npt.NDArray[np.float64], t: float, vol: float) -> npt.NDArray[np.float64]:
    """ lognormal density as mollifier """
    var = vol**2 * t
    p = np.exp(-0.5 * x**2 / var) / np.sqrt(2.0 * np.pi * var)
    p /= np.trapezoid(p, x)
    return p


def roll_forward(p: npt.NDArray[np.float64], x: npt.NDArray[np.float64], dx: float, ts: float, te: float,
                 local_vol, pde_config: PdeConfig) -> npt.NDArray[np.float64]:
    """ Roll the density forward from time ts to te (ts < te) """
    scheme = pdeschemes.scheme(pde_config, ts)
    scheme.local_vol = local_vol
    p_new = scheme.roll_forward(p, x, ts, te, dx)
    return p_new


def shift_forward(x: npt.NDArray[np.float64], p: npt.NDArray[np.float64], tol: float=1e-6) -> npt.NDArray[np.float64]:
    """ Shift the density to match the forward.
        Bad results have been observed when this is turned on for very short expiries.
        We keep it to False at configuration level, but leave it available by choice. """
    ex = np.exp(x)
    pde_forward_m = np.trapezoid(ex * p, x) # If perfect, would be 1.0
    target_forward_m = 1.0
    if np.abs(pde_forward_m - target_forward_m) > tol: # Only shift if beyond tolerance
        shift = np.log(target_forward_m / pde_forward_m)
        x_shifted = x + shift
        p_shifted = np.interp(x, x_shifted, p, left=0.0, right=0.0)
        p_shifted = np.maximum(p_shifted, 0.0)
        pde_forward_m = expectation(ex, p_shifted, x) # If perfect, would be 1.0
        # pde_forward_m = np.trapezoid(ex * p_shifted, x) # If perfect, would be 1.0
        return p_shifted
    else:
        return p


def mass(p_grid: npt.NDArray[np.float64], x_grid: npt.NDArray[np.float64]) -> float:
    """ Calculate mass of the probability density p_grid along variable x_grid
        (i.e. its integral from -infty to +infty) """
    return np.trapezoid(p_grid, x_grid)


def expectation(payoff: npt.NDArray[np.float64], p_grid: npt.NDArray[np.float64],
                x_grid: npt.NDArray[np.float64]) -> float:
    """ Calculate expected value of payoff given probability and x grids """
    return np.trapezoid(payoff * p_grid, x_grid)


def calculate_densities(maturities: npt.NDArray[np.float64], lv, pde_config: PdeConfig) -> dict:
    """ Calculate densities at specified maturities """
    # Initialize spot grid: first maturity if rescaling on x, last maturity otherwise
    spotgrid_tmax = maturities[0] if pde_config.rescale_x else maturities[-1]
    x, dx, spot_idx = build_spotgrid(spotgrid_tmax, pde_config)

    # Initialize density
    start_time = 1.0 / 365.0 # Make sure no payoff is required before that
    p = lognormal_density(x, start_time, pde_config.mesh_vol)

    # Run PDE batches for each maturity
    reports = []
    for mty_idx in range(maturities.shape[0]):
        ts = start_time if mty_idx == 0 else maturities[mty_idx - 1]
        te = maturities[mty_idx]
        step_grid = timegrids.build_timegrid(ts, te, pde_config)
        x, dx, p = density_step(p, x, dx, step_grid, lv, pde_config)

        report = {'start_time': ts, 'end_time': te, 'x_grid': x, 'p_grid': p, 'dx': dx}
        reports.append(report)

    return reports


def price_vanillas(valdate: dt.datetime, expiries: list[dt.datetime], strikes: list[list[float]],
                   fwd_curve: EqForwardCurve, iv_surface: ImpliedVol, lv: LocalVol,
                   **kwargs) -> list[npt.NDArray[np.float64]]:
    """ Helper to price vanillas by PDE on LV process.
        The strikes are a matrix along the expiry direction.
        Return a list of forward prices per expiry, corresponding to each strike.
        The PDE time grid is refined between expiries (the expiries provide the sparse grid).
    """
    n_timesteps = kwargs.get('n_timesteps', 100)
    n_meshes = kwargs.get('n_meshes', 250)
    scheme = kwargs.get('scheme', 'rannacher')
    option_type = string_to_optiontype(kwargs.get('option_type', 'straddle'))

    # Calculate expiry times and forwards
    expiry_times = timegrids.model_time(valdate, expiries)

    # PDE config
    mesh_vol = iv_surface.black_volatility(expiry_times[0], 1.0, 1.0)
    pde_config = PdeConfig(n_timesteps=n_timesteps, n_meshes=n_meshes, mesh_vol=mesh_vol, scheme=scheme,
                           rescale_x=True, rescale_p=True, shift_forward=False,
                           iv_surface=iv_surface)

    # Run PDE to calculate densities at each maturity
    density_reports = calculate_densities(expiry_times, lv.value, pde_config)

    # Calculate PDE prices (forward)
    pde_prices = []
    for r_idx, density_report in enumerate(density_reports):
        dens_mty = density_report['end_time']
        x = density_report['x_grid']
        p = density_report['p_grid']
        fwd = fwd_curve.value(expiries[r_idx])

        # # Check density
        # print(f"Density: {mass(p, x):.6f}")

        # Check timing consistency
        if not isequal(dens_mty, expiry_times[r_idx]):
            raise ValueError(f"Inconsistent times between closed-form and densities at density time {dens_mty}")

        # Calculate PDE prices
        s = fwd * np.exp(x)
        exp_strikes = strikes[r_idx]
        pde_price = []
        for k in exp_strikes: # ToDo: can this be vectorized?
            match option_type:
                case OptionType.CALL:
                    payoff = np.maximum(s - k, 0.0)
                case OptionType.PUT:
                    payoff = np.maximum(k - s, 0.0)
                case OptionType.STRADDLE:
                    payoff = np.abs(s - k)
                case _:
                    raise ValueError(f"Unsupported option type: {option_type}")


            pde_price.append(expectation(payoff, p, x))

        pde_prices.append(np.asarray(pde_price))

    return pde_prices


if __name__ == "__main__":
    print("Hello")

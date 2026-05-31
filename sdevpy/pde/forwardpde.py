import datetime as dt
import numpy as np
import numpy.typing as npt
from sdevpy.pde import pdeschemes
from sdevpy.pde.pdeschemes import PdeConfig
from sdevpy.utilities import timegrids
from sdevpy.utilities.tools import isequal
from sdevpy.market.eqforward import EqForwardCurve
from sdevpy.volatility.localvol.localvol import LocalVol
from sdevpy.instruments.constants import OptionType, string_to_optiontype


FWD_PDE_START_TIME = 1.0 / 365.0 # 1.0 / 52.0


def density_step(old_p: npt.NDArray[np.float64], old_x: npt.NDArray[np.float64], old_dx: float,
                 t_grid: npt.NDArray[np.float64], lv: LocalVol, config: PdeConfig):
    """ Forward PDE evolution along t_grid, with optional rescaling of meshes at
        beginning of the step, optional shifting of the density along its x-axis
        to mach the forward, and optional rescaling of density to integrated to 1
        at end of the step """
    # Rescale spot grid
    if config.rescale_x:
        x, dx, spot_idx = build_spotgrid(t_grid[-1], lv, config)
        p = np.interp(x, old_x, old_p, left=0.0, right=0.0)
    else:
        x, dx, p = old_x, old_dx, old_p

    # Forward reduction
    for i in range(t_grid.shape[0] - 1):
        p = roll_forward(p, x, dx, t_grid[i], t_grid[i + 1], lv, config)

    # Shift to match forward
    if config.shift_forward:
        p = shift_forward(x, p)

    # Rescale density
    if config.rescale_p: # Rescale mass to 1.0 at te
        mass_ = mass(p, x)
        # print(f"Mass: {mass_}")
        p /= mass_

    return x, dx, p


def density(maturity: float, lv: LocalVol, config: PdeConfig):
    """ Simple forward PDE for density, without any rescaling """
    # Build time grid
    t_grid = timegrids.build_timegrid(0.0, maturity, config)
    n_timegrid = t_grid.shape[0]

    # Build spot grid
    x, dx, spot_idx = build_spotgrid(maturity, lv, config)

    # Initial probability
    lnvol = lv.ivol_guess(maturity)
    p = lognormal_density(x, FWD_PDE_START_TIME, lnvol)

    # Forward reduction
    for i in range(n_timegrid - 1):
        p = roll_forward(p, x, dx, t_grid[i], t_grid[i + 1], lv, config)

    return x, dx, p


def build_spotgrid(maturity: float, lv: LocalVol, config: PdeConfig) -> tuple[npt.ArrayLike, float, int]:
    """ Build spot grid for PDEs """
    iv_guess = lv.ivol_guess(maturity) * 1.2 # Conservative factor of 1.2 is common
    # print(f"Mesh vol(build): {iv_guess}")
    n_meshes = config.n_meshes
    x_max = iv_guess * np.sqrt(maturity) * config.n_stdevs
    n_half = int(n_meshes / 2)
    dx = x_max / n_half

    # Vectorized
    x_grid = np.arange(-n_half, n_half + 1) * dx

    # # Old non-vectorized
    # x_grid = np.zeros(2 * n_half + 1)
    # x = 0.0
    # for i in range(n_half):
    #     x = x + dx
    #     x_grid[n_half + 1 + i] = x
    #     x_grid[n_half - 1 - i] = -x

    return x_grid, dx, n_half


def lognormal_density(x: npt.NDArray[np.float64], t: float, vol: float) -> npt.NDArray[np.float64]:
    """ lognormal density as mollifier """
    var = vol**2 * t
    p = np.exp(-0.5 * x**2 / var) / np.sqrt(2.0 * np.pi * var)
    p /= np.trapezoid(p, x)
    return p


def roll_forward(p: npt.NDArray[np.float64], x: npt.NDArray[np.float64], dx: float, ts: float, te: float,
                 local_vol: LocalVol, pde_config: PdeConfig) -> npt.NDArray[np.float64]:
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


def calculate_densities(maturities: npt.NDArray[np.float64], lv: LocalVol, pde_config: PdeConfig) -> dict:
    """ Calculate densities at specified maturities """
    # Initialize spot grid: first maturity if rescaling on x, last maturity otherwise
    spotgrid_tmax = maturities[0] if pde_config.rescale_x else maturities[-1]
    x, dx, spot_idx = build_spotgrid(spotgrid_tmax, lv, pde_config)

    # Initialize density
    start_time = FWD_PDE_START_TIME #/ 10.0 # Make sure no payoff is required before that
    lnvol = lv.ivol_guess(start_time)
    # print(f"Mollifier vol: {lnvol}")
    p = lognormal_density(x, start_time, lnvol)

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


def get_pde_config(t: float=None, **kwargs) -> PdeConfig:
    """ Inspect the kwargs and retrieve PDE config.
        The time argument is only used if the volatility surface is passed too (iv_surface),
        in which case it is used to estimate the mesh vol at ATM.
    """
    pde_config = kwargs.get('pde_config', None)
    if pde_config is None: # Set using the other arguments
        n_timesteps = kwargs.get('n_timesteps', 100)
        n_meshes = kwargs.get('n_meshes', 250)
        scheme = kwargs.get('scheme', 'rannacher')

        pde_config = PdeConfig(n_timesteps=n_timesteps, n_meshes=n_meshes, scheme=scheme,
                               rescale_x=True, rescale_p=True, shift_forward=False)

    return pde_config


def vanilla_expectation(fwd, p, x, strikes, option_type):
    spot = fwd * np.exp(x)
    pde_price = []

    # Vectorized
    k_2d = np.asarray(strikes)[:, None] # (n_strikes, 1)
    match option_type:
        case OptionType.CALL:
            payoffs = np.maximum(spot[None, :] - k_2d, 0.0)
        case OptionType.PUT:
            payoffs = np.maximum(k_2d - spot[None, :], 0.0)
        case OptionType.STRADDLE:
            payoffs = np.abs(spot[None, :] - k_2d)
        case _:
            raise ValueError(f"Unsupported option type: {option_type}")

    pde_price = expectation(payoffs, p, x)

    # # Old non-vectorized
    # for k in strikes:
    #     match option_type:
    #         case OptionType.CALL:
    #             payoff = np.maximum(spot - k, 0.0)
    #         case OptionType.PUT:
    #             payoff = np.maximum(k - spot, 0.0)
    #         case OptionType.STRADDLE:
    #             payoff = np.abs(spot - k)
    #         case _:
    #             raise ValueError(f"Unsupported option type: {option_type}")

    #     pde_price.append(expectation(payoff, p, x))

    return pde_price


def price_vanilla_surface(valdate: dt.datetime, expiries: list[dt.datetime], strikes: list[list[float]],
                          fwd_curve: EqForwardCurve, lv: LocalVol, **kwargs) -> list[npt.NDArray[np.float64]]:
    """ Price a surface of vanillas by PDE on LV process.
        The strikes are a matrix along the expiry direction.
        Return a list of forward prices per expiry, corresponding to each strike.
        The PDE time grid is refined between expiries (the expiries provide the sparse grid).
    """
    option_type = string_to_optiontype(kwargs.get('option_type', 'straddle'))

    # Calculate expiry times
    expiry_times = timegrids.model_time(valdate, expiries)

    # Set PDE config
    pde_config = get_pde_config(expiry_times[0], **kwargs)

    # Run PDE to calculate densities at each maturity
    density_reports = calculate_densities(expiry_times, lv, pde_config)

    # Calculate PDE prices (forward)
    pde_prices = []
    for r_idx, density_report in enumerate(density_reports):
        dens_mty = density_report['end_time']
        x = density_report['x_grid']
        p = density_report['p_grid']
        fwd = fwd_curve.value(expiries[r_idx])

        # Check timing consistency
        if not isequal(dens_mty, expiry_times[r_idx]):
            raise ValueError(f"Inconsistent times between closed-form and densities at density time {dens_mty}")

        # Calculate PDE prices
        exp_strikes = strikes[r_idx]
        pde_price = vanilla_expectation(fwd, p, x, exp_strikes, option_type)
        pde_prices.append(np.asarray(pde_price))

    return pde_prices


def price_vanillas(valdate: dt.datetime, expiry: dt.datetime, strikes: list[float],
                   fwd_curve: EqForwardCurve, lv: LocalVol, **kwargs) -> npt.NDArray[np.float64]:
    """ Price a list of vanillas by PDE on LV process.
        Return a list of forward prices, corresponding to each strike.
        The PDE time grid is sparse grid to expiry, refined between points on the sparse grid.
    """
    option_type = string_to_optiontype(kwargs.get('option_type', 'straddle'))

    # Calculate expiry time
    expiry_time = timegrids.model_time(valdate, expiry)

    # Set PDE config
    pde_config = get_pde_config(expiry_time, **kwargs)

    # Build sparse grid
    start_time = FWD_PDE_START_TIME
    if expiry_time < start_time:
        raise ValueError(f"Numerical method not supported before 1D, used with t = {expiry_time}")

    # Build sparse time grid to run the density steps on
    sparse_timegrid = timegrids.build_sparse_timegrid(expiry_time)
    if len(sparse_timegrid) < 1:
        raise ValueError("Invalid step grid for PDE")

    if not isequal(sparse_timegrid[-1], expiry_time):
        msg = "Invalid sparse time grid, last point not matching maturity"
        raise ValueError(f"{msg}: {sparse_timegrid[-1]}/{expiry_time}")

    # Calculate PDE density at maturity
    dens_report = calculate_densities(sparse_timegrid, lv, pde_config)[-1]
    dens_t, dens_p, dens_x = dens_report['end_time'], dens_report['p_grid'], dens_report['x_grid']

    # Check time consistency
    if not isequal(dens_t, expiry_time):
        msg = "Unexpected time for final density not equal to maturity"
        raise ValueError(f"{msg}: {dens_t}/{expiry_time}")

    # Calculate prices
    fwd = fwd_curve.value(expiry)
    prices = vanilla_expectation(fwd, dens_p, dens_x, strikes, option_type)

    # spot = fwd * np.exp(dens_x)
    # for strike in strikes:
    #     payoff = np.maximum(spot - strike, 0.0) if is_call else np.maximum(strike - spot, 0.0)
    #     prices.append(expectation(payoff, dens_p, dens_x))

    return np.asarray(prices)


if __name__ == "__main__":
    print("Hello")

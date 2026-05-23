import numpy as np
import numpy.typing as npt
from sdevpy.volatility.impliedvol.impliedvol import ImpliedVol
from sdevpy.volatility.localvol.localvol import LocalVol
from sdevpy.volatility.impliedvol.optionsurface import OptionQuoteType
from sdevpy.pde import forwardpde as fpde
from sdevpy.pde.pdeschemes import PdeConfig
from sdevpy.utilities.tools import isequal
from sdevpy.utilities import timegrids


DFLT_PDE_CONFIG = PdeConfig()


########### TODO ################################
# * Pass simpler than PdeConfig by estimating mesh_vol using LV?
#   Impl. generic integrated variance calculation


class NumericalImpliedVol(ImpliedVol):
    def __init__(self, lv: LocalVol, pde_config: PdeConfig=DFLT_PDE_CONFIG, **kwargs):
        super().__init__()
        self.lv = lv
        self.calculate_type = OptionQuoteType.ForwardPremium
        self.pde_config = pde_config

    def calculate(self, t: float, k: npt.ArrayLike, is_call: bool, f: float) -> npt.ArrayLike:
        """ Calculate the forward prices using a numerical method (typically forward PDE) """
        start_time = fpde.FWD_PDE_START_TIME
        if t < start_time:
            raise ValueError(f"Numerical method not supported for use before 1D, used with t = {t}")

        # Build sparse time grid to run the density steps on
        step_grid = timegrids.build_sparse_timegrid(t)
        # print(f"Maturity: {t}")
        # print(f"Step grid: {step_grid}")
        if len(step_grid) < 1:
            raise ValueError("Invalid step grid for PDE")

        if np.abs(step_grid[-1] - t) > self.time_epsilon:
            raise ValueError(f"Invalid PDE time step grid, last point not equal to maturity: {step_grid[-1]}/{t}")

        dens_report = fpde.calculate_densities(step_grid, self.lv.value, self.pde_config)[-1]
        dens_t, dens_p, dens_x = dens_report['end_time'], dens_report['p_grid'], dens_report['x_grid']
        if not isequal(dens_t, t):
            raise ValueError(f"Unexpected time for final density not equal to maturity: {dens_t}/{t}")

        # Calculate prices
        prices = []
        spot = f * np.exp(dens_x)
        for strike in k:
            # print(spot)
            # print(strike)
            payoff = np.maximum(spot - strike, 0.0) if is_call else np.maximum(strike - spot, 0.0)
            prices.append(fpde.expectation(payoff, dens_p, dens_x))

        return np.asarray(prices)

    def dump_data(self) -> dict:
        """ Dump to dictionary """
        result = {'lv': self.lv.dump_data()}
        return result

    # @staticmethod
    # def build_step_grid(t: float, term_tol: float=1.0/52.0) -> list[float]:
    #     """ Construct PDE time step grid with minimum granularity up to the specified cut-off time.
    #         If the last granular point is within term_tol of the cut-off time, it is ignored and
    #         the cut-off time is taken instead. """
    #     if t < 0.0:
    #         raise ValueError(f"Negative time requested in PDE time step grid building: {t}")

    #     coarse_grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    #     if t < coarse_grid[0]:
    #         step_grid = [t]
    #     else:
    #         long_term_size = 1.0 # In years
    #         step_grid = [point for point in coarse_grid if point < t]
    #         current_t = step_grid[-1] + long_term_size
    #         cut_off_years = 100.0
    #         while current_t < t:
    #             step_grid.append(current_t)
    #             current_t += long_term_size
    #             if current_t > cut_off_years:
    #                 raise ValueError("Unexpected large step grid built for PDE")

    #         if step_grid[-1] > t - term_tol:
    #             step_grid[-1] = t
    #         else:
    #             step_grid.append(t)

    #     return np.asarray(step_grid)


if __name__ == "__main__":
    import datetime as dt
    import numpy as np
    from sdevpy.market import eqvolsurface as vsurf
    from sdevpy.market.eqforward import get_forward_curves
    from sdevpy.utilities import timegrids
    from sdevpy.volatility.localvol.lvsection_calib import calibrate_lv_bysections
    from sdevpy.volatility.impliedvol.numerical_impliedvol import NumericalImpliedVol, DFLT_PDE_CONFIG


    name, valdate = "ABC", dt.datetime(2025, 12, 15)

    # Retrieve forward curve
    fwd_curve = get_forward_curves([name], valdate)[0]

    # Retrieve option data
    file = vsurf.data_file(name, valdate)
    option_data = vsurf.eqvolsurfacedata_from_file(file)
    mkt_data = {'option_data': option_data, 'forward_curve': fwd_curve}
    print(f"Retrieved market data from file {file}")

    # Access data in object
    expiries = option_data.expiries
    fwds = fwd_curve.value(expiries)
    mkt_strikes = option_data.get_strikes(fwd_curve=fwd_curve, to_type='absolute')
    mkt_vols = option_data.vols

    # Quick check of size consistency
    print(f"Number of expiries: {len(expiries)}")
    print(f"Number of forwards: {len(fwds)}")
    print(f"Number of strike sections: {len(mkt_strikes)}")
    print(f"Number of vol sections: {len(mkt_vols)}")
    for i in range(len(expiries)):
        print(f"Expiry {i+1} number of strikes/vols: {len(mkt_strikes[i])}/{len(mkt_vols[i])}")



    # Choose model
    section_model = 'BiExp' # SVI, CubicVol, BiExp

    # Calibration config
    # lv_data_folder = lvf.test_data_folder()
    config = {'start_new': True, 'model': section_model, 'store_date': valdate, 'optimizer': 'SLSQP',
            'tol': 1e-6, 'pde_timesteps': 50,  'pde_spotsteps': 100, #'lv_folder': lv_data_folder,
            'sol_as_init': False}

    # Calibrate LV
    print("Launching calibration")
    calib_result = calibrate_lv_bysections(valdate, name, config, verbose=True, calc_pde_vols=True)
    lv = calib_result['lv']
    print(lv)


    pde_config = DFLT_PDE_CONFIG
    num_iv = NumericalImpliedVol(lv, pde_config=pde_config)

    # Retrieve data
    surface_data = calib_result['iv_data'] # Get from inputs instead? Better check.
    expiries = surface_data.expiries
    expiry_grid = np.array([timegrids.model_time(valdate, expiry) for expiry in expiries])

    # Retrieve forward curve
    fwd_curve = get_forward_curves([name], valdate)[0]

    # fwds = surface_data.forwards
    fwds = fwd_curve.value(expiries)
    strike_surface = surface_data.get_strikes(fwd_curve=fwd_curve, to_type='absolute')
    vol_surface = surface_data.vols

    is_call = True
    num_iv_prices, num_iv_vols = [], []
    for exp_idx, expiry in enumerate(expiry_grid):
        f = fwds[exp_idx]
        strikes = strike_surface[exp_idx]
        num_iv_prices.append(num_iv.calculate(expiry, strikes, is_call, f))
        num_iv_vols.append(['ToDo'])

    print(f"Num. IV prices: {num_iv_prices}")

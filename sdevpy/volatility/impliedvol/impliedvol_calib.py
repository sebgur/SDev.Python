import logging
import numpy as np
import numpy.typing as npt
from sdevpy.volatility.impliedvol.parametric_impliedvol import ParametricImpliedVol
from sdevpy.maths import constants
from sdevpy.maths.metrics import rmse
from sdevpy.maths.optimization import create_optimizer
from sdevpy.utilities import timegrids
from sdevpy.volatility.impliedvol.optionsurface import OptionQuoteType
log = logging.getLogger(__name__)


class TsIvObjectiveBuilder:
    def __init__(self, model: ParametricImpliedVol, expiries: npt.ArrayLike,
                 strikes: npt.ArrayLike, fwds: npt.ArrayLike, mkt_vols: npt.ArrayLike,
                 mkt_prices: npt.ArrayLike):
        self.model = model
        self.expiries = expiries
        self.strikes = strikes
        self.fwds = fwds
        self.market_vols = mkt_vols
        self.market_prices = mkt_prices
        self.is_call = True

        # Get target values
        self.target_values = None
        match self.model.calculate_type:
            case OptionQuoteType.ForwardPremium:
                self.target_values = self.market_prices
            case OptionQuoteType.LogNormalVol:
                self.target_values = self.market_vols
            case _:
                raise ValueError(f"Calculation type not supported: {self.model.calculate_type}")

    def objective(self, params):
        # Update params
        self.model.update_params(params)
        is_ok, penalty = self.model.check_params()
        if is_ok:
            # Calculate model vols
            model_values = self.model.calculate(self.expiries, self.strikes, self.is_call, self.fwds)

            # Calculate the objective function
            obj = rmse(model_values, self.target_values)
            return obj
        else:
            # In principle we should return a penalty number. However, it is not clear at the moment if
            # that penalty should come from the model (where we know the parameters) or the
            # objective function (where we know the problem). It might need to come from both.
            # For now we are using a problem-specific penalty, i.e. the value if all the model prices
            # were 0, assuming that should be much bigger than at any reasonable solution.
            # return 100.0 * self.target_values.sum()
            return constants.FLOAT_INFTY # Didn't this cause problems before?


class TsIvCalibrator:
    """ Calibrates a ParametricImpliedVol with a global optimizer """
    def __init__(self, model: ParametricImpliedVol, config: dict):
        self.model = model
        self.config = config
        self.times, self.strikes, self.fwds = None, None, None
        self.mkt_vols, self.mkt_prices = None, None
        self.result = None
        self.sol = None

    # def calibrate(self, mkt_data: EqVolSurfaceData, init_point=None) -> None:
    def calibrate(self, mkt_data: dict, init_point=None) -> None:
        # Retrieve target data
        self.prepare_target_data(mkt_data)

        # Set model's initial state
        init_params = (init_point if init_point is not None else self.model.initial_point())
        self.model.update_params(init_params)

        # Optimizer settings
        method = self.config.get('optimizer', 'SLSQP')
        tol = self.config.get('tol', 1e-6)

        # Constraints
        bounds = self.model.bounds()

        # Objective
        builder = TsIvObjectiveBuilder(self.model, self.times, self.strikes, self.fwds,
                                       self.mkt_vols, self.mkt_prices)
        objective = builder.objective

        # Optimize
        optimizer = create_optimizer(method, tol=tol)
        self.result = optimizer.minimize(objective, x0=init_params, bounds=bounds)
        self.sol = self.result.x # Optimum parameters

        # Warn if failure to converge
        if not self.result.success:
            log.warning(f"Optimization did not converge: {self.result.message}")

        # Make sure model is left at solution point
        self.model.update_params(self.sol)

    def prepare_target_data(self, mkt_data: dict) -> None:# EqVolSurfaceData) -> None:
        """ Flatten target market data into lists of values along all expiries """
        fwd_curve = mkt_data['forward_curve']
        option_data = mkt_data['option_data']
        expiries = option_data.expiries
        fwds = fwd_curve.value(expiries)
        strike_surface = option_data.get_strikes(fwd_curve=fwd_curve, to_type='absolute')
        vol_surface = option_data.vols
        price_surface = option_data.get_prices(fwd_curve, option_type='call') # Just calls for now

        # Reformat inputs to flat vectors
        valdate = option_data.valdate
        self.times, self.strikes, self.fwds = [], [], []
        self.mkt_vols, self.mkt_prices = [], []
        for i in range(len(expiries)):
            expiry = timegrids.model_time(valdate, expiries[i])
            fwd = fwds[i]
            strikes = strike_surface[i]
            vols = vol_surface[i]
            prices = price_surface[i]
            for strike, vol, price in zip(strikes, vols, prices, strict=True):
                self.times.append(expiry)
                self.fwds.append(fwd)
                self.strikes.append(strike)
                self.mkt_vols.append(vol)
                self.mkt_prices.append(price)

        self.times = np.asarray(self.times)
        self.strikes = np.asarray(self.strikes)
        self.fwds = np.asarray(self.fwds)
        self.mkt_vols = np.asarray(self.mkt_vols)
        self.mkt_prices = np.asarray(self.mkt_prices)
        self.model.base_date = valdate

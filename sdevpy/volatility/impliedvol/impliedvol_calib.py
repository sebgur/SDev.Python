import numpy as np
import numpy.typing as npt
from sdevpy.volatility.impliedvol.zerosurface import ParametricZeroSurface
from sdevpy.maths.metrics import rmse
from sdevpy.market.eqvolsurface import EqVolSurfaceData
from sdevpy.maths.optimization import create_optimizer
from sdevpy.tools import timegrids
from sdevpy.volatility.impliedvol.optionsurface import (OptionTarget, keep_positive,
    check_expiries_and_forwards, convert_to_target_values, OptionQuoteType)


class TsIvObjectiveBuilder:
    def __init__(self, model: ParametricZeroSurface, expiries: npt.ArrayLike,
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
            return 100.0 * self.target_values.sum()


class TsIvCalibrator:
    """ Calibrates a ParametricZeroSurface with a global optimizer """
    def __init__(self, model: ParametricZeroSurface, config: dict):
        self.model = model
        self.config = config
        self.times, self.strikes, self.fwds = None, None, None
        self.mkt_vols, self.mkt_prices = None, None
        self.result = None
        self.sol = None

    def calibrate(self, mkt_data: EqVolSurfaceData, init_point=None) -> None:
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

        # Make sure model is left at solution point
        self.model.update_params(self.sol)

    def prepare_target_data(self, mkt_data: EqVolSurfaceData) -> None:
        """ Flatten target market data into lists of values along all expiries """
        expiries = mkt_data.expiries
        fwds = mkt_data.forwards
        strike_surface = mkt_data.get_strikes('absolute')
        vol_surface = mkt_data.vols
        price_surface = mkt_data.call_prices # Just calls for now

        # Reformat inputs to flat vectors
        valdate = mkt_data.valdate
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

    def check_consistency(self, options: list[list[OptionTarget]]) -> list[list[OptionTarget]]:
        """ Take out negative rate options depending on model features.
            Check consistency of expiries, forwards, etc.
            Note: this function is not used yet, but will be in a later phase if/when
            we want to introduce negative rates. """
        # Strip out negative rate options if needed
        t_options = (options if self.model.allow_negative_variables else keep_positive(options))

        # Check consistency of expiries and forwards
        check_expiries_and_forwards(t_options)

        # Convert from quoted type to targetType required for model calibration.
        c_options = convert_to_target_values(t_options, self.model.calculate_type, self.model.shift)

        return c_options

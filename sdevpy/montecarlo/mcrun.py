import numpy as np
import datetime as dt
from sdevpy.montecarlo.FactorModel import MultiAssetGBM
from sdevpy.montecarlo.PathGenerator import PathGenerator
from sdevpy.montecarlo.payoffs.payoffs import Payoff, VanillaOption
from sdevpy.montecarlo.payoffs.exotics import WorstOfBarrier, BasketOption, AsianOption
from sdevpy.montecarlo.MonteCarloPricer import MonteCarloPricer
from sdevpy.models import localvol_factory as lvf
from sdevpy.analytics import black
from sdevpy.tools import timegrids, timer


#################### TODO #########################################################################
# * Implement Payoff algebra
# * Mistral: in case we lose the page, here was the prompt to create an algebraic structure
#   "How can I create a Domain Specific Language and make composable trees from payoff primitives?"

# * Handle event dates and the interpolation to discretization grid
# * Event date design: the most flexible may be to interpolate the paths out of the path build.
#   Those paths will come together with a certain discretization time grid, which we can assume
#   to be known/extracted from the path builder. Ideally we would want to interpolate knowing
#   the forward curves, but we could also (as a first step at least) just do a linear interpolation.
#   For that, we should do a caching of the interpolation indexes and coefficients.
#   That is, we could define some kind of "path date" object that would be a datetime augmented with
#   the indexes and coefficients for its interpolation along the discretization time grid.

# * Introduce concept of past fixings
# * Implement var swap spread payoff
# * Introduce portfolios, multi-cash-flow payoffs
# * Check accuracy against LV calib
# * Calculate vega through LV calib

# * Implement no-arb time parametric IVs (mixture of lognormals, SVI)
# * Try implementing exact Dupire LV calibration using AAD on BS prices? Or IVs for SVI?
# * Mistral: use numba JIT, parallelization (joblib, Ray)


import numpy as np
from abc import ABC, abstractmethod


class McConfig:
    def __init__(self, **kwargs):
        self.n_time_steps = kwargs.get('n_time_steps', 25)


if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)
    names = ['CalibIndex', 'SPX', 'NKY']
    spot = np.asarray([100, 100, 50])
    drift = np.asarray([0.02, 0.05, 0.04])
    sigma = np.asarray([0.2, 0.3, 0.1])
    n_assets = len(spot)
    df = 0.90

    # Load local vols (3 times the same)
    lv = lvf.load_lv_from_folder(None, valdate, names[0], lvf.test_data_folder())
    lvs = [lv, lv, lv]

    # Correction matrix
    corr = np.array([[1.0, 0.5, 0.1],
                     [0.5, 1.0, 0.1],
                     [0.1, 0.5, 1.0]])

    # Time grid
    T = 1.0
    n_steps = 50
    config = McConfig(n_time_steps=n_steps + 1)
    time_grid = timegrids.build_timegrid(0.0, T, config)

    # MC paths
    n_paths = 100 * 1000
    constr_type = 'incremental'
    constr_type = 'brownianbridge'
    rng_type = 'sobol'

    # Forward curves
    fwd_curves = []
    for s, mu in zip(spot, drift):
        # Use the default variable trick to circumvent late binding in python loops
        # Otherwise, all the lambda functions will effectively be the same
        fwd_curves.append(lambda t, s=s, mu=mu: s * np.exp(mu * t))

    # Model
    path_timer = timer.Stopwatch("Factor paths")
    model = MultiAssetGBM(spot, sigma, lvs, fwd_curves, time_grid)

    generator = PathGenerator(model, time_grid, constr_type=constr_type,
                              rng_type=rng_type, scramble=False, corr_matrix=corr)

    # Generate underlying paths: n_mc x (n_steps + 1) x n_assets
    paths = generator.generate_paths(n_paths)
    path_timer.stop()
    print(f"Number of assets: {n_assets}")
    print(f"Number of simulations: {n_paths}")
    print(f"Number of times points: {n_steps + 1}")
    print(f"Path shape: {paths.shape}")

    # Vanilla
    name = 'CalibIndex'
    strike = 100
    optiontype = 'Call'
    payoff = VanillaOption(name, strike, optiontype)

    # Basket
    # b_names = ['SPX', 'NKY']
    # weights = [0.5, 0.1]
    # strike = 100
    # optiontype = 'Call'
    # payoff = BasketOption(b_names, weights, strike, optiontype)

    # Asian
    # name = 'SPX'
    # strike = 100
    # optiontype = 'Call'
    # payoff = AsianOption(name, strike, optiontype)

    # Worst-Of Barrier
    # b_names = ['SPX', 'NKY']
    # strike = 100
    # barrier = 49
    # optiontype = 'Call'
    # payoff = WorstOfBarrier(b_names, strike, optiontype, barrier)

    # New payoff design
    # S1 = Terminal(0)
    # S2 = Terminal(1)

    # basket = 0.5 * S1 + 0.5 * S2
    # payoff = Maximum(basket - 100, 0)

    payoff.set_pathindexes(names)

    # MC pricer
    mc_timer = timer.Stopwatch('MC')
    mc = MonteCarloPricer(df=df)
    # This is the separation we need. The MC path builder only requires the paths
    # of the underlying assets as a big multi-d vector. This way we could potentially
    # get those paths from an independent engine.
    mc_price = mc.build(paths, payoff)
    mc_timer.stop()

    # Closed-form
    name_idx = 0
    fwd = spot * np.exp(drift * T)
    cf_price = df * black.price(T, strike, optiontype == 'Call', fwd[name_idx], sigma[name_idx])

    print("MC:", mc_price)
    print("CF:", cf_price)
    path_timer.print()
    mc_timer.print()

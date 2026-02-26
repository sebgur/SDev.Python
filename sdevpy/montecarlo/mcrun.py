import numpy as np
import datetime as dt
from sdevpy.montecarlo.FactorModel import MultiAssetGBM
from sdevpy.montecarlo.PathGenerator import PathGenerator
from sdevpy.montecarlo.payoffs.basic import *
from sdevpy.montecarlo.payoffs.vanillas import VanillaOption, VanillaOptionPayoff
from sdevpy.montecarlo.payoffs.exotics import WorstOfBarrier, BasketOption, AsianOption
from sdevpy.montecarlo.MonteCarloPricer import MonteCarloPricer, McConfig
from sdevpy.models import localvol_factory as lvf
from sdevpy.analytics import black
from sdevpy.tools import timegrids, timer


#################### TODO #########################################################################
# * Introduce portfolios
# * UT for MC with all payoffs
# * Handle event dates: check what happens if maturity date (as event) is last on disc. grid
# * Introduce discount curve and discount at cash-flow payment times
# * Implement simple forward curve as linear interpolation of surface's forwards
# * Introduce concept of past fixings
# * Implement var swap spread payoff
# * Introduce multi-cash-flow payoffs
# * Check accuracy against LV calib
# * Calculate vega through LV calib
# * Greeks by saving BM and maybe time interpolation too?
# * Add more runtime measurement granularity, make timer able to add/store elapsed
# * Make entry point of code from portfolio specification

# * Implement no-arb time parametric IVs (mixture of lognormals, SVI)
# * Implementing exact Dupire through python vectorization and try parallelization
# * Implement DE with parallelization on the population
# * Mistral: use numba JIT, parallelization (joblib, Ray)


def get_spots(names, valdate):
    """ Temp function to get the spots. ToDo: replace by proper function """
    mkt_spot_data = {'ABC': 100.0, 'KLM': 100.0, 'XYZ': 50.0}
    spots = np.asarray([mkt_spot_data.get(name, None) for name in names])
    return spots


def get_forward_curves(names, valdate):
    spot = get_spots(names, valdate)
    drift = np.asarray([0.02, 0.05, 0.04])
    fwd_curves = []
    for s, mu in zip(spot, drift):
        # Use the default variable trick to circumvent late binding in python loops
        # Otherwise, all the lambda functions will effectively be the same
        fwd_curves.append(lambda t, s=s, mu=mu: s * np.exp(mu * t))

    return fwd_curves


def get_local_vols(names, valdate, **kwargs):
    folder = kwargs.get('folder', lvf.test_data_folder())
    lvs, sigmas = [], []
    for name in names:
        sigmas.append(0.2)
        lvs.append(lvf.load_lv_from_folder(None, valdate, name, folder))

    return lvs, sigmas


def get_correlations(names, valdate):
    corr = np.array([[1.0, 0.5, 0.1],
                     [0.5, 1.0, 0.1],
                     [0.1, 0.5, 1.0]])
    return corr


def book_currency(book):
    """ Temp to get the book pricing currency, to get the discount curve """
    return "USD"


def get_eventdates(book, valdate):
    d1y = dt.datetime(valdate.year + 1, valdate.month, valdate.day)
    return np.asarray([d1y])


def build_timegrid(valdate, eventdates, config):
    max_date = eventdates.max()
    max_T = timegrids.model_time(valdate, max_date)
    disc_tgrid = timegrids.build_timegrid(0.0, max_T, config)
    return disc_tgrid


if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)

    # Create portfolio
    book = []
    v_name, v_strike, v_type = 'ABC', 100.0, 'Call' # For check against CF
    book.append(VanillaOption(v_name, v_strike, v_type))
    book.append(BasketOption(['XYZ', 'KLM'], [0.5, 0.1], 100.0, 'Call'))
    book.append(AsianOption('ABC', 100.0, 'Call'))
    book.append(WorstOfBarrier(['ABC', 'XYZ'], 100.0, 'Call', 35.0))

    # Gather model information relevant to required assets
    names = list_names(book)
    n_names = len(names)
    print(f"Book names: {names}")
    print(f"Number of assets: {n_names}")

    # Retrieve discount curve, assuming all payoffs in same currency/same CSA
    csa_ccy = book_currency(book)
    df = 0.90

    # Retrieve spots
    spot = get_spots(names, valdate)

    # Retrieve forward curves
    fwd_curves = get_forward_curves(names, valdate)

    # Retrieve local vols
    lvs, sigma = get_local_vols(names, valdate)

    # Retrieve correlations
    corr = get_correlations(names, valdate)

    # MC configuration
    n_paths = 100 * 1000
    constr_type = 'incremental'
    constr_type = 'brownianbridge'
    rng_type = 'sobol'
    n_steps = 50
    config = McConfig(n_time_steps=n_steps + 1)
    print(f"Number of simulations: {n_paths}")
    print(f"Number of times points: {n_steps + 1}")

    # Build time grid
    eventdates = get_eventdates(book, valdate)
    disc_tgrid = build_timegrid(valdate, eventdates, config)

    # Set model
    model = MultiAssetGBM(spot, sigma, lvs, fwd_curves, disc_tgrid)

    # Set spot path generator
    generator = PathGenerator(model, disc_tgrid, constr_type=constr_type,
                              rng_type=rng_type, scramble=False, corr_matrix=corr)

    # Generate spots paths on the discretization grid: n_mc x (n_steps + 1) x n_assets
    timer_path = timer.Stopwatch("Generate spot paths")
    paths = generator.generate_paths(n_paths)
    timer_path.stop()
    print(f"Path shape: {paths.shape}")

    # MC pricer
    timer_mc = timer.Stopwatch('Payoff calculation')
    payoff = book[0]
    payoff.set_nameindexes(names)
    mc = MonteCarloPricer(df=df)

    # This is the separation we need. The MC path builder only requires the paths
    # of the underlying assets as a big multi-d vector. This way we could potentially
    # get those paths from an independent engine.

    # First we project the discretization grid paths on the event date paths before
    # calculating the payoffs, wich require the event date paths.
    # event_paths = mc.interpolate_eventdates(paths, eventdates)

    mc_price = mc.build(paths, payoff)
    timer_mc.stop()

    # Closed-form for vanilla
    event_tgrid = np.array([timegrids.model_time(valdate, date) for date in eventdates])
    T = event_tgrid[-1]
    name_idx = names.index(v_name)
    fwd = fwd_curves[name_idx](T)
    cf_price = df * black.price(T, v_strike, v_type, fwd, sigma[name_idx])

    print("MC:", mc_price)
    print("CF:", cf_price)
    timer_path.print()
    timer_mc.print()

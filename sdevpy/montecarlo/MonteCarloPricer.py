import numpy as np
import datetime as dt
# from sdevpy.montecarlo.payoffs.basic import list_names
from sdevpy.models import localvol_factory as lvf
from sdevpy.tools import timegrids, timer
from sdevpy.montecarlo.FactorModel import MultiAssetGBM
from sdevpy.montecarlo.PathGenerator import PathGenerator


# The MC path builder only requires the paths of the underlying assets as a big
# multi-d vector. This way we can get those paths from an independent engine.

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


def price_book(valdate, book, **kwargs):
    # Gather all names in the book and set their indexes in the instruments
    names = book.names
    # names = list_names([t.instrument for t in book])
    # n_names = len(names)
    # for trade in book:
    #     trade.instrument.set_nameindexes(names)

    # Retrieve discount curve, assuming all payoffs in same currency/same CSA
    csa_curve_id = book.csa_curve_id
    # csa_ccy = book_currency(book)
    df = 0.90

    # Retrieve modelling data
    spot = get_spots(names, valdate)
    fwd_curves = get_forward_curves(names, valdate)
    lvs, sigma = get_local_vols(names, valdate)
    corr = get_correlations(names, valdate)

    # MC configuration
    n_paths = 100 * 1000
    constr_type = 'incremental'
    constr_type = 'brownianbridge'
    rng_type = 'sobol'
    n_steps = 50
    config = McConfig(n_time_steps=n_steps + 1)

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
    # print(f"Path shape: {paths.shape}")

    # MC pricer
    timer_mc = timer.Stopwatch('Payoff calculation')
    mc = MonteCarloPricer(df=df)

    # First we project the discretization grid paths on the event date paths before
    # calculating the payoffs, which only require the event date paths.
    # event_paths = mc.interpolate_eventdates(paths, eventdates)

    mc_price = mc.build(paths, book)
    timer_mc.stop()

    return mc_price


class MonteCarloPricer:
    def __init__(self, df):
        self.df = df

    def build(self, paths, book):
        ids = []
        pvs = []
        for trade in book.trades:
            instr = trade.instrument
            instr_paths = instr.evaluate(paths)
            disc_pvs = self.df * np.mean(instr_paths)
            ids.append(trade.name)
            pvs.append(disc_pvs)

        result = {'id': ids, 'pv': pvs}
        return result


class McConfig:
    def __init__(self, **kwargs):
        self.n_time_steps = kwargs.get('n_time_steps', 25)

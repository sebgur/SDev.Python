""" The MC path builder only requires the paths of the underlying assets as a big
    multi-d vector. This way we can get those paths from an independent engine. """
import numpy as np
import datetime as dt
from sdevpy.models import localvol_factory as lvf
from sdevpy.tools import timegrids, timer
from sdevpy.montecarlo.FactorModel import MultiAssetGBM
from sdevpy.montecarlo.PathGenerator import PathGenerator
from sdevpy.market.spot import get_spots
from sdevpy.market.yieldcurve import get_yieldcurve
from sdevpy.market.eqforward import get_forward_curves
from sdevpy.montecarlo.payoffs.basic import get_eventdates


def get_local_vols(names, valdate, **kwargs):
    folder = kwargs.get('folder', lvf.test_data_folder())
    lvs = []
    for name in names:
        lvs.append(lvf.load_lv_from_folder(None, valdate, name, folder))

    return lvs


def get_correlations(names, valdate):
    corr = np.array([[1.0, 0.5, 0.1],
                     [0.5, 1.0, 0.1],
                     [0.1, 0.5, 1.0]])
    return corr


def build_timegrid(valdate, eventdates, config):
    max_date = eventdates.max()
    max_T = timegrids.model_time(valdate, max_date)
    disc_tgrid = timegrids.build_timegrid(0.0, max_T, config)
    return disc_tgrid


def price_book(valdate, book, **kwargs):
    book.set_nameindexes()
    eventdates = book.get_eventdates(valdate)

    # Retrieve modelling data
    names = book.names
    disc_curve = get_yieldcurve(book.csa_curve_id, valdate)
    spot = get_spots(names, valdate)
    fwd_curves = get_forward_curves(names, valdate)
    lvs = get_local_vols(names, valdate)
    corr = get_correlations(names, valdate)

    # Build time grid
    disc_tgrid = build_timegrid(valdate, eventdates, McConfig(**kwargs))

    # Set model
    model = MultiAssetGBM(spot, fwd_curves, lvs, disc_tgrid)

    # Set spot path generator
    generator = PathGenerator(model, disc_tgrid, **kwargs, corr_matrix=corr)

    # MC pricer
    n_paths = kwargs.get('n_paths', 10 * 1000)
    df = disc_curve.discount(eventdates.max())
    mc = MonteCarloPricer(path_generator=generator, df=df, n_paths=n_paths)

    # First we project the discretization grid paths on the event date paths before
    # calculating the payoffs, which only require the event date paths.

    mc_price = mc.pv(book)
    mc.print_timers()
    return mc_price


class MonteCarloPricer:
    def __init__(self, path_generator, df, n_paths):
        self.path_generator = path_generator
        self.df = df
        self.n_paths = n_paths
        self.timers = None

    def pv(self, book):
        # Generate spots paths on the discretization grid: n_mc x (n_steps + 1) x n_assets
        timer_path = timer.Stopwatch("Generate spot paths")
        paths = self.path_generator.generate_paths(self.n_paths)
        timer_path.stop()

        # Calculate payoffs
        timer_payoff = timer.Stopwatch('Payoff calculation')
        pvs = self.build(paths, book)
        timer_payoff.stop()

        self.timers = [timer_path, timer_payoff]
        return pvs

    def build(self, paths, book):
        ids = []
        pvs = []
        for trade in book.trades:
            instr = trade.instrument
            # In principle we should discount before taking the mean. However,
            # here we can discount after the mean as we only consider deterministic
            # discount rates for now. If rates were to be stochastic, the discounting,
            # i.e. the division by the numeraire, should be done before taking the mean.
            fwd_flows = instr.evaluate(paths)
            mean_fwd_flows = np.mean(fwd_flows)
            disc_pvs = self.df * mean_fwd_flows
            ids.append(trade.name)
            pvs.append(disc_pvs)

        result = {'id': ids, 'pv': pvs}
        return result

    def print_timers(self):
        for timer in self.timers:
            timer.print()


class McConfig:
    """ Monte-Carlo engine config 
        - constr_type: Brownian motion construction (incremental, brownianbridge)
        - rng_type: random number generator (mt, sobol, halton, latinhypercube)
    """
    def __init__(self, **kwargs):
        self.n_timesteps = kwargs.get('n_timesteps', 50)
        # self.n_paths = kwargs.get('n_paths', 10 * 1000)
        # self.constr_type = kwargs.get('constr_type', 'brownianbridge')
        # self.rng_type = kwargs.get('rng_type', 'sobol')
        # self.scramble = kwargs.get('scramble', False)

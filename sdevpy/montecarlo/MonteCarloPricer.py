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
# from sdevpy.montecarlo.payoffs.basic import get_eventdates


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
    names = book.set_nameindexes()
    book.set_valuation_date(valdate)
    eventdates = book.eventdates
    # book.set_eventindexes(eventdates)

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
    event_times = timegrids.model_time(valdate, eventdates)
    # print(f"Event times: {event_times}")
    mc = MonteCarloPricer(generator, n_paths, event_times, df=df)

    # First we project the discretization grid paths on the event date paths before
    # calculating the payoffs, which only require the event date paths.

    mc_price = mc.pv(book)
    mc.print_timers()
    return mc_price


def path_interp_coeffs(disc_times, event_times):
    disc_times = np.asarray(disc_times)
    event_times = np.asarray(event_times)

    # indices of left grid point
    idx = np.searchsorted(disc_times, event_times) - 1
    idx = np.clip(idx, 0, len(disc_times) - 2)

    t_left = disc_times[idx]
    t_right = disc_times[idx + 1]

    w1 = (event_times - t_left) / (t_right - t_left)
    w0 = 1.0 - w1
    return idx, w0, w1


def interp_paths(paths, idx, w0, w1):
    """ Input path shape (n_paths, n_disctimes, n_factors).
        Output path shape (n_paths, n_eventtimes, n_factors) """
    # Path shape: (n_paths, n_times, n_assets)
    S_left = paths[:, idx, :] # broadcasting
    S_right = paths[:, idx + 1, :]

    # Reshape weights for broadcasting
    w0 = w0.reshape(1, -1, 1)
    w1 = w1.reshape(1, -1, 1)

    return w0 * S_left + w1 * S_right

class MonteCarloPricer:
    def __init__(self, path_generator, n_paths, event_times, df):
        self.path_generator = path_generator
        self.disc_times = path_generator.time_grid
        self.event_times = event_times
        self.df = df
        self.n_paths = n_paths
        self.timers = []

    def pv(self, book):
        # Generate spots paths on disc. grid: n_mc x (n_steps + 1) x n_assets
        timer_path = timer.Stopwatch("Generate spot paths")
        paths = self.path_generator.generate_paths(self.n_paths)
        timer_path.stop()

        # Interpolate paths from disc. to event date grid
        timer_interp = timer.Stopwatch("Interpolate to event grid")
        print(f"Disc. path shape: {paths.shape}")
        idx, w0, w1 = path_interp_coeffs(self.disc_times, self.event_times)
        event_paths = interp_paths(paths, idx, w0, w1)
        print(f"Event path shape: {event_paths.shape}")
        timer_interp.stop()

        # Calculate payoffs
        timer_payoff = timer.Stopwatch('Payoff calculation')
        pvs = self.build(paths, book)
        # pvs = self.build(event_paths, book)
        # print(f"Disc. paths: {paths}")
        # print(f"Event paths: {event_paths}")
        timer_payoff.stop()

        self.timers = [timer_path, timer_interp, timer_payoff]
        return pvs

    def build(self, paths, book):
        ids = []
        pvs = []
        for trade in book.trades:
            print(trade.name)
            instr = trade.instrument
            # In principle we should discount before taking the mean. However,
            # here we can discount after the mean as we only consider deterministic
            # discount rates for now. If rates were to be stochastic, the discounting,
            # i.e. the division by the numeraire, should be done before taking the mean.
            fwd_flows = instr.evaluate(paths)
            # print(fwd_flows)
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

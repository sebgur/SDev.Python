import datetime as dt
import logging
import numpy as np
import numpy.typing as npt
from sdevpy.utilities import dates as dts
from sdevpy.maths.interpolation import create_interpolation, Interpolation
from sdevpy.market.fx.fxconventions import (fx_market_yearfraction, conventional_pair_name, parse_fx_pair,
                                            is_premium_adjusted)
from sdevpy.market.fx.fxforward import fx_spot_date, fx_pillar_date
from sdevpy.volatility.fx.fx_deltastrike import bs_delta
log = logging.getLogger(__name__)


class FxVolInterpolation:
    """ Two-dimensional interpolation of the FX vol surface along expiry and delta directions.
        Delta direction: one Interpolation object per expiry pillar.
        Expiry direction: the two surrounding pillar smiles are read at the requested put delta,
                          giving vols s0 at t0 and s1 at t1. Those are then combined linearly in
                          variance, i.e.
                          w(t) = w0 + (t - t0) / (t1 - t0) * (w1 - w0),  w_i = s_i^2 * t_i,
                          and the result is returned as sqrt(w(t) / t).
        Args:
            expiries: pillar expiries (dates)
            time_interp: 'var' interpolates s^2 * t, 'vol2' interpolates s^2, 'vol' interpolates s.
            time_extrap: 'flat' holds the first/last pillar vol outside the pillar range,
                         'linear' extrapolates the variance linearly.
    """
    def __init__(self, valdate: dt.datetime, expiries: list[dt.datetime], moneyness_interps: list[Interpolation],
                 fwds: npt.ArrayLike, pair: str, spot: float, forcurve, domcurve, **kwargs):
        self.pair = pair
        self.forccy, self.domccy = conventional_pair_name(*parse_fx_pair(pair))
        self.spot = spot
        self.forcurve, self.domcurve = forcurve, domcurve
        self._settle_cache = {}
        self.prem_adj = is_premium_adjusted(self.forccy, self.domccy)
        self.spot_delta_cutoff = kwargs.get('spot_delta_cutoff', '1Y')
        self.spot_delta_cutoff_date = fx_pillar_date(valdate, self.spot_delta_cutoff,
                                                     self.forccy, self.domccy)

        time_interp = kwargs.get('time_interp', 'var')
        time_extrap = kwargs.get('time_extrap', 'flat')

        # Check grid consistency
        n_expiries, n_interps, n_fwds = len(expiries), len(moneyness_interps), len(fwds)
        if not n_expiries == n_interps == n_fwds:
            raise ValueError(f"""Incompatible sizes between expiries, interpolations and forwards:
                                 {n_expiries}/{n_interps}/{n_fwds}""")

        if n_expiries == 0:
            raise ValueError("At least one expiry pillar is required")

        # Time interpolation
        self.time_interp = time_interp.lower()
        if self.time_interp not in ('var', 'vol2', 'vol'):
            raise ValueError(f"Unknown time interpolation: {time_interp}")

        self.time_extrap = time_extrap.lower()
        if self.time_extrap not in ('flat', 'linear'):
            raise ValueError(f"Unknown time extrapolation: {time_extrap}")

        # Order pillars by increasing expiry
        order = sorted(range(n_expiries), key=lambda i: expiries[i])
        self.expiries = [expiries[i] for i in order]
        self.interps = [moneyness_interps[i] for i in order]
        self.fwds = np.asarray([fwds[i] for i in order], dtype=float)
        if np.any(self.fwds <= 0.0):
            raise ValueError("Forwards must be strictly positive")

        # Conversion to times
        self.valdate = valdate
        self.times = self._to_times(self.expiries)
        # self.times = [fx_market_yearfraction(self.valdate, expiry) for expiry in self.expiries]
        if np.any(self.times <= 0.0):
            raise ValueError("Expiry pillars must be strictly after the valuation date")

        if np.any(np.diff(self.times) <= 0.0):
            raise ValueError("Duplicate/unordered expiry pillars")

    def vol_at_moneyness(self, expiry: npt.ArrayLike, moneyness: npt.ArrayLike) -> npt.ArrayLike:
        """ Vol at (expiry, log-forward-moneyness m = log(K/F)) """
        t = self._to_times(expiry)
        m = np.asarray(moneyness, dtype=float)
        t, m = np.broadcast_arrays(t, m)
        shape = t.shape
        tf, mf = t.reshape(-1), m.reshape(-1)
        if tf.size == 0:
            return np.empty(shape, dtype=float)

        if np.any(tf <= 0.0):
            raise ValueError("Requested expiries must be strictly after the valuation date")

        i0, i1 = self._brackets(tf)
        s0, s1 = self._smile_values(i0, mf), self._smile_values(i1, mf)
        return self._combine_in_time(tf, i0, i1, s0, s1).reshape(shape)

    def vol_at_strike(self, expiry: npt.ArrayLike, strike: npt.ArrayLike, fwd: npt.ArrayLike=None) -> npt.ArrayLike:
        """ Vol at (expiry, strike), converting once through the delivery-date forward.
            When fwd is None, it is calculated internally. When fwd is provided, it must be the forward
            for the option's delivery date following the expiry. """
        f = self.forward(expiry) if fwd is None else np.asarray(fwd, dtype=float)
        k = np.asarray(strike, dtype=float)
        if np.any(k <= 0.0):
            raise ValueError("Requested strikes must be strictly positive")

        return self.vol_at_moneyness(expiry, np.log(k / f))

    #######################################################################################################
    def vol_at_delta(self, expiry, delta, option_type, iters: int=80,
                     double_root_preference: str='large') -> npt.ArrayLike:
        """ Vol at a MARKET delta quote, e.g. delta=0.25 option_type='C' for a 25-delta call """
        k, t, f = self._solve_delta_strike(expiry, delta, option_type, iters, double_root_preference)
        return self.vol_at_strike(t, k, fwd=f)

    def strike_at_delta(self, expiry, delta, option_type, iters: int=80,
                        double_root_preference: str='large') -> npt.ArrayLike:
        """ Strike of a market delta quote, from the same solve """
        k, _, _ = self._solve_delta_strike(expiry, delta, option_type, iters, double_root_preference)
        return k

    def _solve_delta_strike(self, expiry, delta, option_type, iters: int=80,
                            double_root_preference: str='large', tol: float=1e-14) -> tuple:
        """ Strike whose market delta matches the quote, with the vol read off this surface, i.e.
            the root of g(K) = bs_delta(K, vol_at_strike(K)) - target.

            Bisection needs g to change sign across the bracket. That holds wherever delta is
            monotone in strike: puts, and plain (non premium-adjusted) calls. Premium-adjusted
            call delta is not monotone -- it rises from zero, peaks, and falls back to zero -- so
            the full bracket never straddles the root and a naive bisection silently collapses
            onto an end point. Its peak is strictly below the forward (it solves
            N(d2).sigma.sqrt(T) = n(d2), whose root is positive for any sigma.sqrt(T) < 0.8), so
            [log F, hi] is monotone decreasing and still brackets every target below the
            at-the-forward delta, which covers all normal quotes. Only above that are there zero
            or two roots, and those go to strike_from_delta, which locates the peak by ternary
            search, applies the double-root convention and reports validity. """
        expiries = np.asarray(expiry)
        f, df_f = self._fwd_and_df_f(expiry)
        t = self._to_times(expiry)
        spot_delta = expiries <= self.spot_delta_cutoff_date

        phi = 1.0 if str(option_type).upper().startswith('C') else -1.0
        target = phi * np.abs(np.asarray(delta, dtype=float))
        disc = np.where(spot_delta, df_f, 1.0)
        expiries, t, f, df_f, target, disc, spot_delta = np.broadcast_arrays(
            np.atleast_1d(expiries), np.atleast_1d(t), np.atleast_1d(f), np.atleast_1d(df_f),
            np.atleast_1d(target), np.atleast_1d(disc), np.atleast_1d(spot_delta))

        # Bracket in log-strike, same generous width strike_from_delta uses by default
        width = 15.0 * np.asarray(self.vol_at_strike(t, f, fwd=f)) * np.sqrt(t) + 1.0 # More aggressive, faster
        # width = 15.0 * np.asarray(self.vol_at_strike(t, f, fwd=f)) * np.sqrt(t) + 8.0
        lo, hi = np.log(f) - width, np.log(f) + width

        def g(log_k):
            k = np.exp(log_k)
            return bs_delta(f, k, self.vol_at_strike(t, k, fwd=f), t, phi, disc, self.prem_adj) - target

        log_f = np.log(f)
        g_lo, g_hi, g_f = g(lo), g(hi), g(log_f)

        # Pick a valid bracket per element, or flag it for the delegated solve
        straddles = np.sign(g_lo) != np.sign(g_hi)
        outer_only = ~straddles & (np.sign(g_f) != np.sign(g_hi))
        bracketed = straddles | outer_only
        lo = np.where(outer_only, log_f, lo)

        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            take_low = (g(mid) > 0) == (g_hi > 0)
            hi = np.where(take_low, mid, hi)
            lo = np.where(take_low, lo, mid)
            if np.max(hi - lo) < tol: # log-strike precision is exhausted well before iters
                break

        k = np.exp(0.5 * (lo + hi))

        # Unbracketed points: premium-adjusted calls above the at-the-forward delta
        if not np.all(bracketed):
            k = np.asarray(k).copy()
            sub = ~bracketed
            k[sub] = self._delta_strike_fixed_point(
                expiries[sub], t[sub], f[sub], df_f[sub], target[sub], spot_delta[sub],
                option_type, double_root_preference)

        return k, t, f

    def _delta_strike_fixed_point(self, expiries, t, f, df_f, target, spot_delta, option_type,
                                  double_root_preference: str, tol: float=1e-12,
                                  max_iter: int=100) -> npt.ArrayLike:
        """ Delegated solve for the non-monotone branch. strike_from_delta needs a vol to invert,
            so wrap it in the usual fixed point: trial vol -> strike -> smile vol there -> repeat,
            seeded at the at-the-forward vol. """
        df_d = self.spot * df_f / f
        sigma = np.asarray(self.vol_at_strike(t, f, fwd=f), dtype=float)
        opt = 'C' if str(option_type).upper().startswith('C') else 'P'
        for _ in range(max_iter):
            sol = strike_from_delta(self.valdate, expiries, self.spot, df_f, df_d, sigma, target,
                                    opt, self.prem_adj, spot_delta,
                                    double_root_preference=double_root_preference)
            valid = np.asarray(sol.valid).reshape(-1)
            if not np.all(valid):
                i = int(np.nonzero(~valid)[0][0])
                cap = np.asarray(sol.delta_max_abs).reshape(-1)[i]
                raise ValueError(f"No strike matches delta "
                                 f"{abs(np.asarray(target).reshape(-1)[i]):.4f}{opt} at expiry "
                                 f"{np.asarray(expiries).reshape(-1)[i]}: premium-adjusted call "
                                 f"delta is capped at {cap:.4f} there")

            updated = np.asarray(self.vol_at_strike(t, sol.k, fwd=f), dtype=float)
            move = np.max(np.abs(updated - sigma))
            sigma = updated
            if move < tol:
                return np.asarray(sol.k, dtype=float)

        raise RuntimeError(f"Delta strike solve did not converge in {max_iter} iterations")

    ###################################################################################################

    # def vol_at_delta(self, expiry, delta, option_type, iters: int=80) -> npt.ArrayLike:
    #     """ Vol at a MARKET delta quote, e.g. delta=0.25 option_type='C' for a 25-delta call.
    #         A delta quote pins strike and vol jointly, so solve the single root
    #             g(K) = bs_delta(K, vol_at_strike(K)) - target = 0
    #         by bisection in log-strike: one loop, no solver nested inside it. """
    #     lo, hi, t, f = self._delta_bracket(expiry, delta, option_type, iters)
    #     return self.vol_at_strike(t, np.exp(0.5 * (lo + hi)), fwd=f)

    # def strike_at_delta(self, expiry, delta, option_type, iters: int=80) -> npt.ArrayLike:
    #     """ Strike of a market delta quote, from the same bisection """
    #     lo, hi, _, _ = self._delta_bracket(expiry, delta, option_type, iters)
    #     return np.exp(0.5 * (lo + hi))

    # def _delta_bracket(self, expiry, delta, option_type, iters) -> tuple:
    #     """ Bisect g(K) in log-strike. Returns the final bracket, the times and the forward, so
    #         both callers share one solve and one consistent forward. """
    #     f, df_f = self._fwd_and_df_f(expiry)
    #     t = self._to_times(expiry)
    #     prem_adj = self.prem_adj
    #     spot_delta = np.asarray(expiry) <= self.spot_delta_cutoff_date

    #     phi = 1.0 if str(option_type).upper().startswith('C') else -1.0
    #     target = phi * np.abs(np.asarray(delta, dtype=float))
    #     disc = np.where(spot_delta, df_f, 1.0)
    #     t, f, target, disc = np.broadcast_arrays(np.atleast_1d(t), np.atleast_1d(f),
    #                                              np.atleast_1d(target), np.atleast_1d(disc))

    #     # Bracket in log-strike, same generous width strike_from_delta uses by default
    #     width = 15.0 * np.asarray(self.vol_at_strike(t, f, fwd=f)) * np.sqrt(t) + 8.0
    #     lo, hi = np.log(f) - width, np.log(f) + width

    #     def g(log_k):
    #         k = np.exp(log_k)
    #         return bs_delta(f, k, self.vol_at_strike(t, k, fwd=f), t, phi, disc, prem_adj) - target

    #     g_hi = g(hi)
    #     for _ in range(iters): # orientation-agnostic: handles both wings
    #         mid = 0.5 * (lo + hi)
    #         take_low = (g(mid) > 0) == (g_hi > 0)
    #         hi = np.where(take_low, mid, hi)
    #         lo = np.where(take_low, lo, mid)

    #     return lo, hi, t, f

    # def var_at_delta(self, expiry: npt.ArrayLike, put_delta: npt.ArrayLike) -> npt.ArrayLike:
    #     """ Variance sigma^2 * t at (expiry, put delta) """
    #     t = self._to_times(expiry)
    #     v = self.vol_at_delta(t, put_delta)
    #     return v * v * t

    def var_at_strike(self, expiry, strike, fwd=None) -> npt.ArrayLike:
        t = self._to_times(expiry)
        v = self.vol_at_strike(expiry, strike, fwd=fwd)
        return v * v * t

    def pillar_strikes(self, idx: int) -> npt.ArrayLike:
        return self.fwds[idx] * np.exp(np.asarray(self.interps[idx].x_grid, dtype=float))

    # def forward(self, expiry: npt.ArrayLike) -> npt.ArrayLike:
    #     """ Delivery-date forward S * df_f(T_set) / df_d(T_set) for the given expiry dates """
    #     arr = np.asarray(expiry)
    #     if arr.dtype.kind in 'fiu':
    #         raise TypeError("Cannot derive the forward from a year fraction: pass expiry dates, "
    #                         "or pass fwd explicitly")

    #     settle = [self.settlement(e) for e in arr.reshape(-1).tolist()]
    #     df_f = np.asarray(self.forcurve.discount(settle), dtype=float).reshape(arr.shape)
    #     df_d = np.asarray(self.domcurve.discount(settle), dtype=float).reshape(arr.shape)
    #     return self.spot * df_f / df_d

    def forward(self, expiry: npt.ArrayLike) -> npt.ArrayLike:
        """ Delivery-date forward S * df_f(T_set) / df_d(T_set) for the given expiry dates """
        return self._fwd_and_df_f(expiry)[0]

    def _fwd_and_df_f(self, expiry: npt.ArrayLike) -> tuple:
        """ (forward, foreign discount factor) at the option's delivery date. The delta solver
            needs df_f on its own for the spot-delta convention, and it comes free with the
            forward, off the same settlement date. """
        arr = np.asarray(expiry)
        if arr.dtype.kind in 'fiu':
            raise TypeError("Cannot derive the forward from a year fraction: pass expiry dates, "
                            "or pass fwd explicitly")

        settle = [self.settlement(e) for e in arr.reshape(-1).tolist()]
        df_f = np.asarray(self.forcurve.discount(settle), dtype=float).reshape(arr.shape)
        df_d = np.asarray(self.domcurve.discount(settle), dtype=float).reshape(arr.shape)
        return self.spot * df_f / df_d, df_f


    def settlement(self, expiry: dt.datetime) -> dt.datetime:
        """ Delivery date of an option expiring on `expiry`: the spot date after expiry, exactly
            as fx_option_dates derives it at calibration. Cached, because fx_spot_date walks the
            calendar one business day at a time. """
        settle = self._settle_cache.get(expiry)
        if settle is None:
            settle = fx_spot_date(expiry, self.forccy, self.domccy)
            self._settle_cache[expiry] = settle

        return settle

    def check_forwards(self, tol: float=1e-10) -> bool:
        """ The stored pillar forwards define the moneyness axis. If the curves passed in do not
            reproduce them, the axis and the queries disagree and every lookup is shifted. """
        recomputed = self.forward(np.asarray(self.expiries, dtype=object))
        bad = np.abs(recomputed - self.fwds) > tol
        if np.any(bad):
            for i in np.nonzero(bad)[0]:
                log.warning(f"Pillar {self.expiries[i]}: stored forward {self.fwds[i]} vs "
                            f"recomputed {recomputed[i]}")

        return not bool(np.any(bad))

    def vol_grid(self, expiries: npt.ArrayLike, strikes: npt.ArrayLike) -> npt.ArrayLike:
        """ Vols on the full (expiry, strike) grid. Expiries must be dates. """
        e = np.atleast_1d(np.asarray(expiries))
        k = np.atleast_1d(np.asarray(strikes, dtype=float))
        f = self.forward(e)
        return self.vol_at_strike(e[:, None], k[None, :], fwd=f[:, None])

    # def vol_grid(self, expiries: npt.ArrayLike, deltas: npt.ArrayLike) -> npt.ArrayLike:
    #     """ Vols on the full (expiry, delta) grid, shape (n_expiries, n_deltas) """
    #     t = np.atleast_1d(self._to_times(expiries))
    #     d = np.atleast_1d(np.asarray(deltas, dtype=float))
    #     return self.vol_at_delta(t[:, None], d[None, :])

    def calendar_check(self, moneyness: npt.ArrayLike=None) -> bool:
        """ At each FIXED log-forward-moneyness, total variance must be non-decreasing in expiry
            (Gatheral): that is the no-calendar-arbitrage condition. Defaults to the union of the
            calibrated pillar moneyness grids. """
        if moneyness is None:
            moneyness = np.unique(np.concatenate([np.asarray(i.x_grid, dtype=float)
                                                  for i in self.interps]))

        m = np.atleast_1d(np.asarray(moneyness, dtype=float))
        t = np.asarray(self.times)
        w = self.vol_at_moneyness(t[:, None], m[None, :]) ** 2 * t[:, None]
        bad = np.diff(w, axis=0) <= 0.0
        if np.any(bad):
            for i, j in zip(*np.nonzero(bad), strict=True):
                log.warning(f"Calendar arbitrage between pillars {i} and {i + 1} at log-moneyness {m[j]:.4f}")

        return not bool(np.any(bad))

    def _brackets(self, t: npt.ArrayLike) -> tuple:
        """ Indices of the pillars surrounding each t, clipped to the pillar range """
        n = len(self.times)
        if n == 1:
            z = np.zeros(np.shape(t), dtype=int)
            return z, z

        i1 = np.clip(np.searchsorted(self.times, t, side='left'), 1, n - 1)
        return i1 - 1, i1

    def _smile_values(self, idx: npt.ArrayLike, d: npt.ArrayLike) -> npt.ArrayLike:
        """ Vols at deltas d, each point read off the smile of its own pillar idx. Loops over the
            distinct pillars touched rather than over points, so the delta interpolation stays
            vectorized: one call per pillar on the sub-vector that needs it. """
        out = np.empty(d.shape, dtype=float)
        for k in np.unique(idx):
            mask = (idx == k)
            out[mask] = np.asarray(self.interps[k].value(d[mask]), dtype=float).reshape(-1)
        return out

    def _to_times(self, expiries: npt.ArrayLike) -> npt.ArrayLike:
        """ Year fractions from valdate """
        arr = np.asarray(expiries)
        if arr.dtype.kind in 'fiu':
            return arr.astype(float)

        flat = [fx_market_yearfraction(self.valdate, e) for e in arr.reshape(-1).tolist()]
        return np.asarray(flat, dtype=float).reshape(arr.shape)

    def _combine_in_time(self, tf: npt.ArrayLike, i0: npt.ArrayLike, i1: npt.ArrayLike,
                         s0: npt.ArrayLike, s1: npt.ArrayLike) -> npt.ArrayLike:
        """ Combine the two surrounding pillar vols into the vol at each requested time """
        if np.all(i0 == i1): # single pillar surface: flat in time
            v = s0
        else:
            t0, t1 = self.times[i0], self.times[i1]
            theta = (tf - t0) / (t1 - t0)
            if self.time_interp == 'var':
                w = s0 * s0 * t0 + theta * (s1 * s1 * t1 - s0 * s0 * t0)
                v = np.sqrt(np.maximum(w, 0.0) / tf)
            elif self.time_interp == 'vol2':
                v = np.sqrt(np.maximum(s0 * s0 + theta * (s1 * s1 - s0 * s0), 0.0))
            else: # vol
                v = s0 + theta * (s1 - s0)

            # Before the first pillar, always extrapolate as flat. After the last pillar,
            # extrapolate as flat or follow on the chosen interpolation.
            v = np.where(tf < self.times[0], s0, v)
            if self.time_extrap == 'flat':
                v = np.where(tf > self.times[-1], s1, v)

        return v

def interpolation_from_fxvol_data(vol_data: dict, md_prov, **kwargs) -> FxVolInterpolation:
    """ Build the surface interpolation straight from the calibrated data as returned by
        CalibrationDataFileProvider.get_fxvol_data """
    smile_interp = kwargs.get('smile_interp', 'pchip') # pchip, akima, cubicspline, linear
    smile_extrap = kwargs.get('smile_extrap', 'flat') # builtin, flat, use for both left and right
    time_interp = kwargs.get('time_interp', 'var') # var, vol2, vol
    time_extrap = kwargs.get('time_extrap', 'flat') # flat, linear

    pair = vol_data['pair']
    valdate = dt.datetime.strptime(vol_data['date'], dts.DATE_FILE_FORMAT)
    forccy, domccy = conventional_pair_name(*parse_fx_pair(pair))
    spot = md_prov.get_fx_spot(forccy, domccy, valdate)
    forcurve = md_prov.get_xccycurve(forccy, valdate)
    domcurve = md_prov.get_xccycurve(domccy, valdate)
    spot_delta_cutoff = vol_data['spot_delta_cutoff']

    expiries, interps, fwds = [], [], []
    for report in vol_data['tenor_reports']:
        if 'fwd' not in report:
            raise KeyError(f"Tenor report {report.get('tenor')} has no 'fwd'")

        fwd = float(report['fwd'])
        moneyness = np.log(np.asarray(report['strikes'], dtype=float) / fwd)
        interps.append(create_interpolation(interp=smile_interp, l_extrap=smile_extrap, r_extrap=smile_extrap,
                                            x_grid=moneyness, y_grid=report['vols']))
        expiries.append(dt.datetime.strptime(report['expiry'], dts.DATE_FILE_FORMAT))
        fwds.append(fwd)

    return FxVolInterpolation(valdate, expiries, interps, fwds, pair, spot, forcurve, domcurve,
                              time_interp=time_interp, time_extrap=time_extrap, spot_delta_cutoff=spot_delta_cutoff)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sdevpy.market.fileprovider import MarketDataFileProvider
    from sdevpy.calibration.fileprovider import CalibrationDataFileProvider
    from sdevpy.volatility.fx.fx_deltastrike import strike_from_delta

    pair = "USDJPY"
    valdate = dt.datetime(2025, 12, 15)

    # Retrieve calibrated data and build the two-dimensional interpolation
    md_prov = MarketDataFileProvider()
    cal_prov = CalibrationDataFileProvider()
    vol_data = cal_prov.get_fxvol_data(pair, valdate)
    data_sections = vol_data['tenor_reports']
    smile_interp = 'linear' # pchip, akima, cubicspline, linear
    smile_extrap = 'flat' # builtin, flat
    time_interp = 'var' # var, vol2, vol
    time_extrap = 'flat' # flat, linear
    surface = interpolation_from_fxvol_data(vol_data, md_prov, smile_interp=smile_interp, smile_extrap=smile_extrap,
                                            time_interp=time_interp, time_extrap=time_extrap)
    surface.calendar_check()

    print(f"Expiries: {surface.expiries}")

    #### Check smile direction ####
    # Smile interpolation/extrapolation at chosen expiry pillar (assuming reports are time-ordered for simplicity)
    disp_expiry_idx = 3
    expiry = surface.expiries[disp_expiry_idx]
    print(f"Viewing smile at expiry: {expiry}")
    data_section = data_sections[disp_expiry_idx]
    data_strikes = np.asarray(data_section['strikes'])
    # data_deltas = data_section['put_deltas']
    data_vols = data_section['vols']
    disp_strikes  = np.linspace(data_strikes.min()*0.90, data_strikes.max()*1.20, 200)
    # disp_deltas = np.linspace(0.01, 0.99, 200)
    vols_at_strikes = surface.vol_at_strike(expiry, disp_strikes)
    # print("vols_at_strikes", vols_at_strikes)
    plt.plot(disp_strikes, surface.vol_at_strike(expiry, disp_strikes), color='green', label='smile interpolation')
    plt.scatter(data_strikes, data_vols, color='black', label='data')
    plt.legend(loc='upper right')
    plt.show()

    #### Check time direction ####
    e0, e1, e2, en = surface.expiries[0], surface.expiries[2], surface.expiries[3], surface.expiries[-1]
    m = 1.25
    v0, v1, v2, vn = surface.vol_at_moneyness(e0, m), surface.vol_at_moneyness(e1, m), \
                     surface.vol_at_moneyness(e2, m), surface.vol_at_moneyness(en, m)

    print(f"{e0}/{v0}")
    print(f"{e1}/{v1}")
    print(f"{e2}/{v2}")
    print(f"{en}/{vn}")

    # Before first pillar, check flat case
    dm1 = dt.datetime(2025, 12, 20)
    print(f"Before first pillar: {dm1}/{surface.vol_at_moneyness(dm1, m)}")

    # Between two pillars, check variance case
    date_t = dt.datetime(2026, 1, 14)
    vol_t = surface.vol_at_moneyness(date_t, m)
    t1, t2 = fx_market_yearfraction(valdate, e1), fx_market_yearfraction(valdate, e2)
    t = fx_market_yearfraction(valdate, date_t)
    var_1, var_2 = v1**2 * t1, v2**2 * t2
    var_t = var_1 + (var_2 - var_1) / (t2- t1) * (t - t1)
    vol_t_check = np.sqrt(var_t / t)
    print(f"Interp/check: {vol_t}/{vol_t_check}")

    # After last pillar
    print(f"Far vol: {surface.vol_at_moneyness(dt.datetime(2050, 1, 1), m)}")

    #### Round-trip vol_at_strike and vol_at_delta ####
    delta, option_type = 0.30, "C"
    vol_from_delta = surface.vol_at_delta(expiry, delta, option_type)[0]
    print(f"vol_from_delta: {vol_from_delta}")
    spot = surface.spot
    settlement = fx_spot_date(expiry, surface.forccy, surface.domccy)
    df_f = surface.forcurve.discount(settlement)
    df_d = surface.domcurve.discount(settlement)
    prem_adjusted, spot_delta = surface.prem_adj, True
    strike = strike_from_delta(valdate, expiry, spot, df_f, df_d, vol_from_delta, delta, option_type,
                               prem_adjusted, spot_delta).k
    print(f"strike from delta: {strike}")
    print(f"vol from strike: {surface.vol_at_strike(expiry, strike)}")

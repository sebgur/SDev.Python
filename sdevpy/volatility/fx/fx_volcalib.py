""" Show examples of definitions of FX market for vols, delta-strike inversion and interpolation """
import datetime as dt
import numpy as np
import logging
from sdevpy.utilities import dates as dts
from sdevpy.market.provider import MarketDataProvider
from sdevpy.market.fileprovider import MarketDataFileProvider
from sdevpy.market.fx import fxconventions
from sdevpy.market.fx.fxforward import fx_pillar_date
from sdevpy.market.fx.fxvolsurface import wingvols_from_butterfly, fx_option_dates
from sdevpy.volatility.fx.fx_vannavolga import wingvols_from_market_strangle_vv, VannaVolgaSmile
from sdevpy.volatility.fx.fx_deltastrike import strike_from_delta, atm_strike
from sdevpy.utilities import timer
log = logging.getLogger(__name__)


################## TODO ###########################################################################
# * Do the ntdr() change
# * Do the branch change (Claude's performance-notes.md)
# * Do the calendar caching change
# * Order the strikes/deltas in the output of the calibrator
# * Implement the direct spline, flat outside the last deltas
# * Output data object from calibrator to json into calibration data location
# * Create interpolation object from calibration data location
# * See if we can improve speed by multi-threading on tenor (optionally)
# * Implement object that interpolates the spline results across time
# * Move yieldcurves to calib data provider
# * Use delta inversion and illustrate it
# * Clarify the choice of small vs large double-root


class FxVolCalibrator:
    def __init__(self, pair: str, md_prov: MarketDataProvider,
                 extra_deltas=(0.05, 0.01), tail_method: str='exact', delta_tol: float=1e-4):
        self.pair = pair
        self.md_prov = md_prov
        self.extra_deltas = tuple(extra_deltas) if extra_deltas else ()
        self.tail_method = tail_method
        self.delta_tol = delta_tol

        # Set null values
        self.forccy, self.domccy = None, None
        self.forcurve, self.domcurve = None, None
        self.vol_date, self.prem_adj = None, None
        self.market_strangle_quote, self.spot_delta_cutoff = False, '1Y'
        self.date, self.spot = None, None

        # Set pair conventions
        self._set_conventions()

    def calibrate(self, date: dt.date) -> dict:
        """ Calibrate on a certain date """
        log.debug(f"<><><><> Calibrating {self.pair} vol surface on {date.strftime(dts.DATETIME_FORMAT)} <><><><>")

        # Fetch market data
        self._fetch_market_data(date)

        # Calibrate at all tenors
        tenor_results = []
        for tenor_idx in range(len(self.vol_data.tenors)):
            result = self.calibrate_tenor(tenor_idx)
            tenor_results.append(result)

        return {'tenor_reports': tenor_results}

    def calibrate_tenor(self, tenor_idx: int) -> dict:
        """ Calibrate at the given tenor """
        ten_timer = timer.Stopwatch(tenor_idx)
        ten_timer.trigger()
        valdate = self.date

        # Extract raw market data
        tenor = self.vol_data.tenors[tenor_idx]
        atm_vol = self.vol_data.atm_vols[tenor_idx]
        quoted_deltas = self.vol_data.deltas[tenor_idx]
        rrs = self.vol_data.rr[tenor_idx]
        bfs = self.vol_data.bf[tenor_idx]
        print(f"Calibrating tenor: {tenor}")
        print(f"ATM vol: {atm_vol}")
        print(f"Deltas: {quoted_deltas}")
        print(f"RRs: {rrs}")
        print(f"BFs: {bfs}")

        # Calculate discount factors and forward
        expiry, settlement = fx_option_dates(valdate, tenor, self.forccy, self.domccy)
        df_f, df_d = self.forcurve.discount(settlement), self.domcurve.discount(settlement)
        fwd = self.spot * df_f / df_d
        print(f"Foreign df: {df_f}")
        print(f"Domestic df: {df_d}")
        print(f"Forward: {fwd}")

        # Build the full set of market points: every quoted delta level, both wings
        spot_delta = expiry <= self.spot_delta_cutoff_date
        deltas, strikes, vols = [], [], []
        pillars = {}
        for delta, rr, bf in zip(quoted_deltas, rrs, bfs, strict=True):
            if self.market_strangle_quote:
                vol_p, vol_c = wingvols_from_market_strangle_vv(valdate, expiry, self.spot, df_f, df_d,
                                                                atm_vol, rr, bf, delta,
                                                                prem_adjusted=self.prem_adj,
                                                                spot_delta=spot_delta)
            else:
                vol_p, vol_c = wingvols_from_butterfly(atm_vol, rr, bf)

            k_put = strike_from_delta(valdate, expiry, self.spot, df_f, df_d, vol_p, -delta, 'P',
                                      self.prem_adj, spot_delta).k
            k_call = strike_from_delta(valdate, expiry, self.spot, df_f, df_d, vol_c, delta, 'C',
                                       self.prem_adj, spot_delta, double_root_preference='large').k
            pillars[delta] = (k_put, vol_p, k_call, vol_c)
            deltas += [-delta, delta]
            strikes += [k_put, k_call]
            vols += [vol_p, vol_c]

        # Concatenate with ATM
        k_atm = atm_strike(valdate, expiry, fwd, atm_vol, self.prem_adj)
        strikes.append(k_atm)
        vols.append(atm_vol)
        deltas.append(0.50) # Add ATM

        # Calculate additional market vols far in the tails for future extrapolation
        if self.extra_deltas:
            tail_d = [d for d in self.extra_deltas
                      if not np.any(np.isclose(d, quoted_deltas, rtol=0.0, atol=self.delta_tol))]

            if len(tail_d) < len(self.extra_deltas):
                skipped = [d for d in self.extra_deltas if d not in tail_d]
                log.debug(f"{tenor}: extra deltas already quoted, taken from the market: {skipped}")

            if tail_d:
                # Build smile for outermost quoted delta
                d_out = np.min(quoted_deltas)
                k_put, vol_p, k_call, vol_c = pillars[d_out]
                smile = VannaVolgaSmile(valdate=valdate, expiry_dt=expiry, fwd=fwd, k_put=k_put, k_atm=k_atm,
                                        k_call=k_call, vol_put=vol_p, atm_vol=atm_vol, vol_call=vol_c,
                                        extrapolation='none', spot=self.spot, df_f=df_f, df_d=df_d,
                                        prem_adjusted=self.prem_adj, spot_delta=spot_delta)

                for d in tail_d:
                    seed_p, seed_c = (vol_p, vol_c) if d < d_out else (atm_vol, atm_vol)
                    v_p, k_p = self._vol_at_delta(smile, expiry, df_f, df_d, d, False, seed_p, spot_delta)
                    v_c, k_c = self._vol_at_delta(smile, expiry, df_f, df_d, d, True, seed_c, spot_delta)
                    deltas += [-d, d]
                    strikes += [k_p, k_c]
                    vols += [v_p, v_c]

        # Order by increasing deltas/strikes


        ten_timer.stop()
        ten_timer.print()

        report = {'expiry': expiry, 'settlement': settlement,
                  'deltas': deltas, 'strikes': strikes, 'vols': vols}
        return report

    # Fixed point iteration. ToDo: check if we don't already have it and move to a more suitable place if any.
    def _vol_at_delta(self, smile, expiry, df_f, df_d, delta, is_call, seed, spot_delta,
                      tol: float=1e-10, max_iter: int=100) -> tuple:
        """ (vol, strike) at the given unsigned delta. The strike needs the vol and the vol needs the
            strike, so iterate: seed a vol, solve its strike, read the smile there, repeat. Seeding at
            the nearest wing vol rather than ATM converges in a few passes this far out. """
        sigma = float(seed)
        signed = delta if is_call else -delta
        kwargs = {'double_root_preference': 'large'} if is_call else {}
        kwargs['spot_delta'] = spot_delta
        for _ in range(max_iter):
            sol = strike_from_delta(self.date, expiry, self.spot, df_f, df_d, sigma, signed,
                                    'C' if is_call else 'P', self.prem_adj, **kwargs)
            if not bool(np.all(sol.valid)):
                raise ValueError(f"No valid strike at delta={delta} for trial vol={sigma}")
            k = float(sol.k)
            sigma_new = float(smile.vol(k, self.tail_method))
            if not np.isfinite(sigma_new):
                raise ValueError(f"VV extrapolation not arbitrage-free at delta={delta}, K={k}")
            if abs(sigma_new - sigma) < tol:
                return sigma_new, k
            sigma = sigma_new
        raise RuntimeError(f"Tail vol did not converge at delta={delta}")


    def _set_conventions(self) -> None:
        """ Set market convention for pair """
        # Check currency pair
        ccy1, ccy2 = fxconventions.parse_fx_pair(self.pair)
        self.forccy, self.domccy = fxconventions.conventional_pair_name(ccy1, ccy2)
        if self.pair != self.forccy + self.domccy:
            raise ValueError(f"Requested pair {self.pair} not in conventional order")

        # Find premium adjusted
        self.prem_adj = fxconventions.is_premium_adjusted(self.forccy, self.domccy)

    def _fetch_market_data(self, date: dt.date) -> None:
        """ Fetch market data on given date """
        self.date = date

        # Fetch spot
        self.spot = self.md_prov.get_fx_spot(self.forccy, self.domccy, self.date)
        log.debug(f"Spot: {self.spot}")

        # Fetch rate curves
        self.forcurve = self.md_prov.get_xccycurve(self.forccy, self.date)
        self.domcurve = self.md_prov.get_xccycurve(self.domccy, self.date)

        # Fetch vol
        self.vol_data = self.md_prov.get_fx_vol_data(self.pair, self.date)
        # self.vol_data.pretty_print()

        # Others
        self.market_strangle_quote = self.vol_data.market_strangle_quote
        self.spot_delta_cutoff = self.vol_data.spot_delta_cutoff
        self.spot_delta_cutoff_date = fx_pillar_date(self.date, self.spot_delta_cutoff, self.forccy, self.domccy)


if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    # Choose test case
    pair = "USDJPY"
    valdate = dt.datetime(2025, 12, 15)

    # Get market data provider
    md_prov = MarketDataFileProvider()

    # Create calibrator
    calibrator = FxVolCalibrator(pair, md_prov)
    print(f"prem_adjusted: {calibrator.prem_adj}")

    # Calibrate
    cal_timer = timer.Stopwatch('calibrate')
    cal_timer.trigger()
    report = calibrator.calibrate(valdate)
    cal_timer.stop()
    # print(report)

    # Check results
    check_strikes, check_vols = 0.0, 0.0
    for r in report['tenor_reports']:
        check_strikes += np.asarray(r['strikes']).mean()
        check_vols += np.asarray(r['vols']).mean()

    print(f"Check strikes: {check_strikes}")
    print(f"Check vols: {check_vols}")
    strike_ref = 907.2903549319075
    vol_ref = 0.622253494336979
    print(f"Strike status: {'OK' if abs(check_strikes - strike_ref) < 1e-8 else 'FAIL'}")
    print(f"Vol status: {'OK' if abs(check_vols - vol_ref) < 1e-10 else 'FAIL'}")

    # Timer
    cal_timer.print()

    # Plot
    # plt.plot(strikes, vols, label='Interpolation', color='blue')
    # plt.scatter(market_strikes, market_vols, label='Market', color='red', zorder=5)
    # plt.show()

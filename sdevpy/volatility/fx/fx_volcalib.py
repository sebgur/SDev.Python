""" Show examples of definitions of FX market for vols, delta-strike inversion and interpolation """
import datetime as dt
import numpy as np
import logging
from sdevpy.market import fxspot
from sdevpy.utilities import dates as dts
from sdevpy.market.fileprovider import MarketDataFileProvider
from sdevpy.volatility.fx.fx_vannavolga import wingvols_from_market_strangle_vv, VannaVolgaSmile
from sdevpy.market.fxvolsurface import wingvols_from_butterfly
from sdevpy.volatility.fx.fx_deltastrike import strike_from_delta, atm_strike, fx_market_yearfraction
from sdevpy.market.provider import MarketDataProvider
log = logging.getLogger(__name__)


################## TODO ###########################################################################
# * Generate: spot_fwd_cutoff, expiry date
# * Implement the vv-based calculation of extrapolated deltas
# * Implement the direct spline, flat outside the last deltas
# * Implement object that interpolates the spline results across time
# * Measure runtime: date conversion to yearfrac in many places including in strike_from_delta solver
# * Move yieldcurves to calib data provider
# * Use delta inversion and illustrate it


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
        self.vol_date, self.prem_adjusted = None, None
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
        expiry = self.vol_data.expiries[tenor_idx] # ToDo: generate
        df_f, df_d = self.forcurve.discount(expiry), self.domcurve.discount(expiry)
        fwd = self.spot * df_f / df_d
        print(f"Foreign df: {df_f}")
        print(f"Domestic df: {df_d}")
        print(f"Forward: {fwd}")

        # Build the full set of market points: every quoted delta level, both wings
        deltas, strikes, vols = [], [], []
        pillars = {}
        for delta, rr, bf in zip(quoted_deltas, rrs, bfs, strict=True):
            if self.vol_data.market_strangle_quote:
                vol_p, vol_c = wingvols_from_market_strangle_vv(valdate, expiry, self.spot, df_f, df_d,
                                                                atm_vol, rr, bf, delta,
                                                                prem_adjusted=self.prem_adjusted)
            else:
                vol_p, vol_c = wingvols_from_butterfly(atm_vol, rr, bf)

            k_put = strike_from_delta(valdate, expiry, self.spot, df_f, df_d, vol_p, -delta, 'P', self.prem_adj).k
            k_call = strike_from_delta(valdate, expiry, self.spot, df_f, df_d, vol_c, delta, 'C', self.prem_adj).k
            pillars[delta] = (k_put, vol_p, k_call, vol_c)
            deltas += [-delta, delta]
            strikes += [k_put, k_call]
            vols += [vol_p, vol_c]

        # Concatenate with ATM
        k_atm = atm_strike(valdate, expiry, fwd, atm_vol)
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
                t = fx_market_yearfraction(self.date, expiry)
                smile = VannaVolgaSmile(fwd=fwd, expiry=t, k_put=k_put, k_atm=k_atm, k_call=k_call,
                                        vol_put=vol_p, atm_vol=atm_vol, vol_call=vol_c, extrapolation='none',
                                        spot=self.spot, df_f=df_f, df_d=df_d, prem_adjusted=self.prem_adj)

                for d in tail_d:
                    seed_p, seed_c = (vol_p, vol_c) if d < d_out else (atm_vol, atm_vol)
                    v_p, k_p = self._vol_at_delta(smile, expiry, df_f, df_d, d, False, seed=seed_p)
                    v_c, k_c = self._vol_at_delta(smile, expiry, df_f, df_d, d, True, seed=seed_c)
                    deltas += [-d, d]
                    strikes += [k_p, k_c]
                    vols += [v_p, v_c]

        # Order by increasing deltas/strikes

        report = {'deltas': deltas, 'strikes': strikes, 'vols': vols}
        return report

    # Fixed point iteration. ToDo: check if we don't already have it and move to a more suitable place if any.
    def _vol_at_delta(self, smile, expiry, df_f, df_d, delta, is_call, seed,
                      tol: float=1e-10, max_iter: int=100) -> tuple:
        """ (vol, strike) at the given unsigned delta. The strike needs the vol and the vol needs the
            strike, so iterate: seed a vol, solve its strike, read the smile there, repeat. Seeding at
            the nearest wing vol rather than ATM converges in a few passes this far out. """
        sigma = float(seed)
        signed = delta if is_call else -delta
        kwargs = {'double_root_preference': 'large'} if is_call else {}
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
        ccy1, ccy2 = fxspot.parse_fx_pair(self.pair)
        self.forccy, self.domccy = fxspot.conventional_pair_name(ccy1, ccy2)
        if self.pair != self.forccy + self.domccy:
            raise ValueError(f"Requested pair {self.pair} not in conventional order")

        # Find premium adjusted
        self.prem_adj = fxspot.is_premium_adjusted(self.forccy, self.domccy)

    def _fetch_market_data(self, date: dt.date) -> None:
        """ Fetch market data on given date """
        self.date = date

        # Fetch spot
        self.spot = md_prov.get_fx_spot(self.forccy, self.domccy, self.date)
        log.debug(f"Spot: {self.spot}")

        # Fetch rate curves
        self.forcurve = md_prov.get_xccycurve(self.forccy, self.date)
        self.domcurve = md_prov.get_xccycurve(self.domccy, self.date)

        # Fetch vol
        self.vol_data = md_prov.get_fx_vol_data(self.pair, self.date)
        # self.vol_data.pretty_print()


if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    # Choose test case
    pair = "USDJPY"
    valdate = dt.datetime(2025, 12, 15)

    # Get market data provider
    md_prov = MarketDataFileProvider()

    # Create calibrator
    calibrator = FxVolCalibrator(pair, md_prov)
    print(f"prem_adjusted: {calibrator.prem_adjusted}")

    # Calibrate
    report = calibrator.calibrate(valdate)
    print(report)

    # Check results

    # Plot
    # plt.plot(strikes, vols, label='Interpolation', color='blue')
    # plt.scatter(market_strikes, market_vols, label='Market', color='red', zorder=5)
    # plt.show()

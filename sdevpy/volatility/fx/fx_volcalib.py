""" Show examples of definitions of FX market for vols, delta-strike inversion and interpolation """
import datetime as dt
import numpy as np
import logging
import matplotlib.pyplot as plt
from sdevpy.market import fxspot
from sdevpy.utilities import dates as dts
from sdevpy.market.fileprovider import MarketDataFileProvider
from sdevpy.volatility.fx.fx_vannavolga import wingvols_from_market_strangle_vv
from sdevpy.market.fxvolsurface import wingvols_from_butterfly
from sdevpy.volatility.fx.fx_deltastrike import strike_from_delta, atm_strike
from sdevpy.market.provider import MarketDataProvider
log = logging.getLogger(__name__)


################## TODO ###########################################################################
# * Move yieldcurves to calib data provider
# * Implement the direct spline, flat outside the last deltas, but keep the number of deltas/points generic
# * Implement the vv-based calculation of extrapolated deltas
# * Implement a calibration flow that, given the raw data, generates a "calibrated" surface that contains
#   more deltas and the direct wing vols to save calibration time (and possibly interpolation definition)
# * Implement object that interpolates the spline results across time
# * Use delta inversion and illustrate it


class FxVolCalibrator:
    def __init__(self, pair: str, md_prov: MarketDataProvider):
        self.pair = pair
        self.md_prov = md_prov

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
        # Extract raw market data
        tenor = self.vol_data.tenors[tenor_idx]
        atm_vol = self.vol_data.atm_vols[tenor_idx]
        deltas = self.vol_data.deltas[tenor_idx]
        rr = self.vol_data.rr[tenor_idx]
        bf = self.vol_data.bf[tenor_idx]
        print(f"Calibrating tenor: {tenor}")
        print(f"ATM vol: {atm_vol}")
        print(f"Deltas: {deltas}")
        print(f"RRs: {rr}")
        print(f"BFs: {bf}")

        # Calculate discount factors and forward
        expiry = self.vol_data.expiries[view_expiry_idx] # ToDo: generate
        df_f, df_d = self.forcurve.discount(expiry), self.domcurve.discount(expiry)
        fwd = self.spot * df_f / df_d
        print(f"Foreign df: {df_f}")
        print(f"Domestic df: {df_d}")
        print(f"Forward: {fwd}")

        # Build the full set of market points: every quoted delta level, both wings
        market_strikes, market_vols = [], []
        for d, r, b in zip(deltas, rr, bf, strict=True):
            if self.vol_data.market_strangle_quote:
                vol_p, vol_c = wingvols_from_market_strangle_vv(self.date, expiry, self.spot, df_f, df_d,
                                                                atm_vol, r, b, d, prem_adjusted=self.prem_adjusted)
            else:
                vol_p, vol_c = wingvols_from_butterfly(atm_vol, r, b)

            k_put = strike_from_delta(valdate, expiry, self.spot, df_f, df_d, vol_p, -d, 'P').k
            k_call = strike_from_delta(valdate, expiry, self.spot, df_f, df_d, vol_c, d, 'C').k
            market_strikes += [k_put, k_call]
            market_vols += [vol_p, vol_c]

        # Concatenate with ATM
        k_atm = atm_strike(valdate, expiry, fwd, atm_vol)
        market_strikes.append(k_atm)
        market_vols.append(atm_vol)

        # Calculate additional market vols far in the tails for future extrapolation
        return {'strikes': market_strikes, 'vols': market_vols}

    def _set_conventions(self) -> None:
        """ Set market convention for pair """
        # Check currency pair
        ccy1, ccy2 = fxspot.parse_fx_pair(self.pair)
        self.forccy, self.domccy = fxspot.conventional_pair_name(ccy1, ccy2)
        if self.pair != self.forccy + self.domccy:
            raise ValueError(f"Requested pair {self.pair} not in conventional order")

        # Find premium adjusted

        # Find spot-forward cutoff

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
    # Choose test case
    pair = "USDJPY"
    valdate = dt.datetime(2025, 12, 15)
    view_expiry_idx = 0

    # Get market data provider
    md_prov = MarketDataFileProvider()

    # Create calibrator
    calibrator = FxVolCalibrator(pair, md_prov)
    print(f"prem_adjusted: {calibrator.prem_adjusted}")

    # Build the full set of market points: every quoted delta level, both wings, plus ATM
    market_strikes, market_vols = [], []
    for d, r, b in zip(deltas, rr, bf, strict=True):
        if data.market_strangle_quote:
            vol_p, vol_c = fx_vannavolga.wingvols_from_market_strangle_vv(valdate, expiry, spot, df_f, df_d, atm_vol, r, b,
                                                                        delta=d)
        else:
            vol_p, vol_c = wingvols_from_butterfly(atm_vol, r, b)

        k_put = float(strike_from_delta(valdate, expiry, spot, df_f, df_d, vol_p, -d, 'P').k)
        k_call = float(strike_from_delta(valdate, expiry, spot, df_f, df_d, vol_c, d, 'C').k)
        market_strikes += [k_put, k_call]
        market_vols += [vol_p, vol_c]

    # t = fx_market_yearfraction(valdate, expiry)
    k_atm = atm_strike(valdate, expiry, fwd, atm_vol)
    market_strikes.append(k_atm)
    market_vols.append(atm_vol)

    # Build interpolated smile
    delta_idx = 0
    plot_delta, plot_rr, plot_bf = deltas[delta_idx], rr[delta_idx], bf[delta_idx]
    s = fx_vannavolga.smile_from_quotes(valdate, expiry, spot=spot, df_f=df_f, df_d=df_d,
                                        atm_vol=atm_vol, rr=plot_rr, bf=plot_bf, delta=plot_delta)
    strikes = np.linspace(0.9 * fwd, 1.1 * fwd, 50)
    vols = []
    for strike in strikes:
        vols.append(s.vol(strike)) #float(s.vol(k, 'first_order')


    # Plot
    plt.plot(strikes, vols, label='Interpolation', color='blue')
    plt.scatter(market_strikes, market_vols, label='Market', color='red', zorder=5)
    plt.show()

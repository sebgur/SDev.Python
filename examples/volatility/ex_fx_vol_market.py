""" Show examples of definitions of FX market for vols, delta-strike inversion and interpolation """
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sdevpy.market import fxspot
from sdevpy.market.fxvolsurface import fxvolsurfacedata_from_file
from sdevpy.utilities import dates as dts
from sdevpy.utilities import timegrids
from sdevpy.market.fileprovider import MarketDataFileProvider
from sdevpy.volatility.fx import fx_vannavolga
from sdevpy.market.fxvolsurface import wingvols_from_butterfly
from sdevpy.volatility.fx.fx_deltastrike import strike_from_delta
from sdevpy import logger
logger.configure(module_display='partial')


################## TODO ###########################################################################
# * Do it twice and compare strangle vs butterfly results
# * Implement the direct spline (flat outside 25s). Maybe use a quick build_smile() in terms of strikes
#   to get the market strangle to butterfly conversion, and then define the interpolation in terms of
#   deltas.
# * Implement the completed spline (using exact VV to create 10D and 5D, make those choosable)
# * Use delta inversion and illustrate it


# Choose test case
pair = "USDJPY"
valdate = dt.datetime(2025, 12, 15)
view_expiry_idx = 0

# Get market data provider
provider = MarketDataFileProvider()

# Check currency pair
ccy1, ccy2 = fxspot.parse_fx_pair(pair)
forccy, domccy = fxspot.conventional_pair_name(ccy1, ccy2)
if pair != forccy + domccy:
    raise ValueError(f"Requested pair {pair} not in conventional order")

# Fetch data object
md_path = provider.root
file = md_path / "fxoptions" / pair / (valdate.strftime(dts.DATE_FILE_FORMAT) + ".json")
print(f"Looking for file: {file}")
data = fxvolsurfacedata_from_file(file)
# data.pretty_print()

# Retrieve raw data at chosen expiry
expiries = data.expiries
expiry = expiries[view_expiry_idx]
atm_vol = data.atm_vols[view_expiry_idx]
deltas = data.deltas[view_expiry_idx]
rr = data.rr[view_expiry_idx]
bf = data.bf[view_expiry_idx]
print(f"Viewing expiry: {expiry}")
print(f"ATM vol: {atm_vol}")
print(f"Deltas: {deltas}")
print(f"RRs: {rr}")
print(f"BFs: {bf}")

# Retrieve spot
spot = provider.get_fx_spot(forccy, domccy, valdate)
print(f"Spot: {spot}")

# Expiry time
t = timegrids.model_time(valdate, expiry)
print(f"Expiry time: {t}")

# Retrieve rate curves
forcurve = provider.get_xccycurve(forccy, valdate)
domcurve = provider.get_xccycurve(domccy, valdate)
df_for = forcurve.discount(expiry)
df_dom = domcurve.discount(expiry)
print(f"Foreign df: {df_for}")
print(f"Domestic df: {df_dom}")
r_for, r_dom = -np.log(df_for) / t, -np.log(df_dom) / t
print(f"Foreign rate: {r_for}")
print(f"Domestic rate: {r_dom}")
fwd = spot * df_for / df_dom
print(f"Forward: {fwd}")

# Build the full set of market points: every quoted delta level, both wings, plus ATM
market_strikes, market_vols = [], []
for d, r, b in zip(deltas, rr, bf, strict=True):
    if data.market_strangle_quote:
        vol_put, vol_call = fx_vannavolga.wingvols_from_market_strangle_vv(spot, r_dom, r_for, t, atm_vol, r, b, delta=d)
    else:
        vol_put, vol_call = wingvols_from_butterfly(atm_vol, r, b)

    k_put = float(strike_from_delta(spot, r_dom, r_for, t, vol_put, -d, 'P').k)
    k_call = float(strike_from_delta(spot, r_dom, r_for, t, vol_call, d, 'C').k)
    market_strikes += [k_put, k_call]
    market_vols += [vol_put, vol_call]

k_atm = fx_vannavolga.atm_dns_strike(fwd, atm_vol, t)
market_strikes.append(k_atm)
market_vols.append(atm_vol)

# Build interpolated smile
delta_idx = 0
plot_delta, plot_rr, plot_bf = deltas[delta_idx], rr[delta_idx], bf[delta_idx]
s = fx_vannavolga.smile_from_quotes(spot=spot, r_d=r_dom, r_f=r_for, expiry=t, atm_vol=atm_vol, rr=plot_rr, bf=plot_bf,
                                    delta=plot_delta)
strikes = np.linspace(0.9 * fwd, 1.1 * fwd, 50)
vols = []
for strike in strikes:
    vols.append(s.vol(strike)) #float(s.vol(k, 'first_order')


# Plot
plt.plot(strikes, vols, label='Interpolation', color='blue')
plt.scatter(market_strikes, market_vols, label='Market', color='red', zorder=5)
plt.show()




# Write examples for strike inversion and round-trip

# Write examples for volatility interpolation in Vanna-Volga and Spline


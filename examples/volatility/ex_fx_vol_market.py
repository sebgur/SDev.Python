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


################## TODO ###########################################################################
# * Show charts in strike/deltas
# * Use delta inversion and illustrate it
# * Retrieve the pillar call/put vols on the original deltas
# * Do it twice and compare strangle vs butterfly results
# * Implement the direct spline (flat outside 25s). Maybe use a quick build_smile() in terms of strikes
#   to get the market strangle to butterfly conversion, and then define the interpolation in terms of
#   deltas.
# * Implement the completed spline (using exact VV to create 10D and 5D, make those choosable)


# Choose test case
pair = "USDJPY"
valdate = dt.datetime(2025, 12, 15)
view_expiry_idx = 2

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


# Calculate
fwd = spot * df_for / df_dom
print(f"Forward: {fwd}")


# Build smile
s = fx_vannavolga.smile_from_quotes(spot=spot, r_d=r_dom, r_f=r_for, expiry=t, atm_vol=atm_vol, rr=-0.01, bf=0.025)
strikes = np.linspace(0.9 * fwd, 1.1 * fwd, 50)
vols = []
for strike in strikes:
    vols.append(s.vol(strike)) #float(s.vol(k, 'first_order')

plt.plot(strikes, vols)
plt.show()


# Write examples for strike inversion and round-trip

# Write examples for volatility interpolation in Vanna-Volga and Spline


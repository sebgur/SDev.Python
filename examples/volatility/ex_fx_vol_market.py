""" Show examples of definitions of FX market for vols, delta-strike inversion and interpolation """
import datetime as dt
from sdevpy.market import fxspot
from sdevpy.market.fxvolsurface import fxvolsurfacedata_from_file
from sdevpy.utilities import dates as dts
from sdevpy.market.fileprovider import MarketDataFileProvider


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


# Retrieve rate curves


spot = 1.10
r_d, r_f = 0.04, 0.02
expiry = 1.0
rr, bf = 0.01, 0.0025


# Build smile
# s = smile_from_quotes(spot=1.10, r_d=0.04, r_f=0.02, expiry=1.0, atm_vol=0.10, rr=-0.01, bf=0.0025)
# print(f"Pillars: K={s.k_put:.4f}/{s.k_atm:.4f}/{s.k_call:.4f} vol={s.vol_put:.4f}/{s.atm_vol:.4f}/{s.vol_call:.4f}")
# for k in np.linspace(0.95, 1.35, 9):
#     print(f"  K={k:.4f}  vv={float(s.vol(k)):.6f} 1st-order={float(s.vol(k, 'first_order')):.6f}")

# Write examples for strike inversion and round-trip

# Write examples for volatility interpolation in Vanna-Volga and Spline


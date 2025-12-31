import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sdevpy.models.impliedvol import ParamSection
from sdevpy.maths import constants


class SviVolSection(ParamSection):
    def __init__(self):
        super().__init__(svivol_formula)

    def check_params(self):
        return svivol_check_params(self.params)


def svivol(t, x, *params):
    """ SVI-like formula but applied to the vol directly. This no longer has
        the original features of no-arbitrage and the interpretation as a limit
        of Heston. It is purely used for its parametric shape here.
    """
    # Retrieve parameters
    if (len(params) != 5):
        raise RuntimeError(f"Incorrect parameter size in SviVol: {len(params)}")

    a = params[0]
    b = params[1]
    rho = params[2]
    m = params[3]
    sigma = params[4]

    # Check constraints
    is_ok, _ = svivol_check_params(params)
    if not is_ok:
        raise RuntimeError("Invalid SviVol parameters")

    # Calculate
    xm = x - m # x is the log-moneyness
    vol = a + b * (rho * xm + np.sqrt(xm**2 + sigma**2))
    if np.any(vol < 0.0):
        raise ValueError("Negative variance in SVI formula")

    return vol


def svivol_check_params(params):
    a = params[0]
    b = params[1]
    rho = params[2]
    m = params[3]
    sigma = params[4]

    is_ok = True
    # Check constraints
    if a < 0.0 or b < 0.0 or abs(rho) >= 1 or sigma < 0.0:
        is_ok = False

    if is_ok:
        if a + b * sigma * np.sqrt(1.0 - rho**2) < 0.0:
            is_ok = False

    # Sudden death for now
    penalty = 0.0 if is_ok else constants.FLOAT_INFTY

    return is_ok, penalty


def svivol_formula(t, x, params):
    """ Wrapper on SVI formula to take parameter vector as input """
    return svivol(t, x, *params)


def sample_params(t, vol):
    """ Guess parameters for display or optimization initial point """
    a = vol
    b = 0.1 / t
    rho = -0.25
    m = 0.0
    sigma = 0.25 * t
    return np.array([a, b, rho, m, sigma])


def generate_sample_data(valdate, terms, base_vol=0.25,
                         percents=[0.10, 0.25, 0.50, 0.75, 0.90]):
    spot, r, q = 100.0, 0.04, 0.02

    expiries, fwds, strike_surface, vol_surface = [], [], [], []
    for term in terms:
        expiry = valdate + dt.timedelta(days=int(term * 365.25))
        fwd = spot * np.exp((r - q) * term)
        base_std = base_vol * np.sqrt(term)
        a, b, rho, m, sigma = base_vol, 0.1 / term, -0.25, 0.0, 0.25 * term # a, b, rho, m, sigma
        strikes, vols = [], []
        for p in percents:
            logm = -0.5 * base_std**2 + base_std * norm.ppf(p)
            strikes.append(fwd * np.exp(logm))
            # a, b, rho, m, sigma = 0.179, 5.3, -0.40, -0.019, 0.025
            vols.append(svivol(term, logm, a, b, rho, m, sigma))

        expiries.append(expiry)
        fwds.append(fwd)
        strike_surface.append(strikes)
        vol_surface.append(vols)

    return np.array(expiries), np.array(fwds), np.array(strike_surface), np.array(vol_surface)


if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)
    terms = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    base_vol = 0.25
    percents = [0.10, 0.25, 0.50, 0.75, 0.90]

    expiries, fwds, strikes, vols = generate_sample_data(valdate, terms, base_vol, percents)
    n_rows, n_cols = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            exp_idx = n_cols * i + j
            ax.plot(strikes[exp_idx], vols[exp_idx], color='red')
            ax.set_title(expiries[exp_idx])
            ax.set_xlabel('strike')
            ax.set_ylabel('vol')
            # ax.legend()

    fig.suptitle('Option prices, PDE vs CF', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

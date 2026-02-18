""" Bi-exponential model, piecing together two exponential models with smooth junction.
    Each exponential branch takes the shape of the Rebonato parametric form, and we impose
    the first derivative's continuity by constraining the left slope.
    In its current form, this model has 6 free parameters.

    Note: one could ensure second derivative continuity with further constraints, thereby
    reducing the number of free parameters from 6 to 5.

    Intuitive way of reducing parameters:
        * 6 -> 5: make taul = taur to have same asymptotic speed
        * 5 -> 4: fix tau to correspond to a certain percentile
     """
import datetime as dt
import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
from sdevpy.models.impliedvol import ParamSection
from sdevpy.maths import constants


DEATH_PENALTY = 1e6


def create_section(time, param_config=None, fill_sample=True):
    section = BiExpSection(time)
    if param_config is None and fill_sample:
        params = sample_params(time) # Fill with sample
    else:
        params = []
        params.append(param_config['f0'])
        params.append(param_config['fl'])
        params.append(param_config['fr'])
        params.append(param_config['taul'])
        params.append(param_config['taur'])
        params.append(param_config['fp'])

    section.update_params(params)
    return section


class BiExpSection(ParamSection):
    def __init__(self, time):
        super().__init__(time, biexp_formula)
        self.model = 'BiExp'

    def check_params(self):
        return biexp_check_params(self.params)

    def dump_params(self):
        data = {'f0': self.params[0], 'fl': self.params[1], 'fr': self.params[2],
                'taul': self.params[3], 'taur': self.params[4], 'fp': self.params[5]}
        return data

    def constraints(self):
        # f0, fl, fr, taul, taur, fp
        lw_bounds = [0.01, 0.01, 0.01, 0.01, 0.01, -2.0]
        up_bounds = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        bounds = opt.Bounds(lw_bounds, up_bounds, keep_feasible=False)
        return bounds


def biexp(x, *params):
    """ BiExp formula. Currently centered at 0 but could be centered somewhere else. """
    # Center
    m = 0.0

    # Retrieve parameters
    if (len(params) != 6):
        raise RuntimeError(f"Incorrect parameter size in BiExp: {len(params)}")

    f0 = params[0]
    fl = params[1]
    fr = params[2]
    taul = params[3]
    taur = params[4]
    fp = params[5]

    # Check constraints
    is_ok, _ = biexp_check_params(params)
    if not is_ok:
        raise RuntimeError("Invalid BiExp parameters")

    # Calculate
    xm = x - m

    # Right case
    ar = fp + (f0 - fr) / taur
    volr = fr + (ar * xm + f0 - fr) * np.exp(-xm / taur)

    # Left case
    al = fp - (f0 - fl) / taul
    voll = fl + (al * xm + f0 - fl) * np.exp(xm / taul)

    return np.where(xm >= 0.0, volr, voll)


def biexp_check_params(params):
    is_ok = False
    if len(params) == 6:
        f0 = params[0]
        fl = params[1]
        fr = params[2]
        taul = params[3]
        taur = params[4]
        fp = params[5]

        is_ok = True
        # Check constraints
        if fl < 0.0 or f0 < 0.0 or fr < 0.0 or taul < 0.0 or taur < 0.0:
            is_ok = False

        if fl < f0: # Not sure this is really a constraint we should impose
            is_ok = False

        # Might want to impose positivity contraints here

    # Sudden death for now
    penalty = 0.0 if is_ok else DEATH_PENALTY
    return is_ok, penalty


def biexp_formula(t, x, params):
    """ Wrapper on BiExp formula to take parameter vector as input """
    return biexp(x, *params)


def sample_params(t, vol=0.25):
    """ Guess parameters for display or optimization initial point """
    f0 = vol
    fl = vol + 0.05
    fr = vol + 0.03
    taul = 0.5 * vol * np.sqrt(t)
    taur = 0.25 * vol * np.sqrt(t)
    fp = 0.0
    return np.array([f0, fl, fr, taul, taur, fp])


def generate_sample_data(valdate, terms, base_vol=0.25,
                         percents=[0.10, 0.25, 0.50, 0.75, 0.90]):
    spot, r, q = 100.0, 0.04, 0.02

    expiries, fwds, strike_surface, vol_surface = [], [], [], []
    for term in terms:
        expiry = valdate + dt.timedelta(days=int(term * 365.25))
        fwd = spot * np.exp((r - q) * term)
        base_std = base_vol * np.sqrt(term)
        params = sample_params(term, base_vol)
        strikes, vols = [], []
        for p in percents:
            logm = -0.5 * base_std**2 + base_std * norm.ppf(p)
            strikes.append(fwd * np.exp(logm))
            vols.append(biexp(logm, *params))

        expiries.append(expiry)
        fwds.append(fwd)
        strike_surface.append(strikes)
        vol_surface.append(vols)

    return np.array(expiries), np.array(fwds), np.array(strike_surface), np.array(vol_surface)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    valdate = dt.datetime(2025, 12, 15)
    terms = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    base_vol = 0.25
    percents = [0.10, 0.25, 0.50, 0.75, 0.90]

    params = [0.25, 0.30, 0.28, 1.0, 1.5, 0.0]
    x = np.linspace(-5, 5, 100)
    v = biexp(x, *params)
    plt.plot(x, v)
    plt.show()

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

    fig.suptitle('Vols', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

import numpy as np
import scipy.optimize as opt
from sdevpy.models.impliedvol import ParamSection
from sdevpy.maths.constants import DEATH_PENALTY


def create_section(time, param_config=None, fill_sample=True):
    section = CubicVolSection(time)
    if param_config is None and fill_sample:
        params = sample_params(time) # Fill with sample
    else:
        params = []
        params.append(param_config['atm'])
        params.append(param_config['skew'])
        params.append(param_config['kurt'])
        params.append(param_config['vl'])
        params.append(param_config['vr'])

    section.update_params(params)
    return section


class CubicVolSection(ParamSection):
    def __init__(self, time):
        super().__init__(time, cubicvoleps_formula)
        self.model = 'CubicVol'
        self.eff_params = None

    def value(self, t, x):
        return self.formula(t, x, self.eff_params)

    def update_params(self, new_params):
        """ We optimize on the original parameters which have a more intuitive meaning, but it is
            more computationally efficient to use the epsilons inside """
        self.params = new_params.copy()
        is_ok, penalty, epsl, epsr = cubicvol_check_params(self.params)
        if is_ok:
            eff_params = self.params.copy()
            eff_params[3] = epsl
            eff_params[4] = epsr
            self.eff_params = eff_params
        else:
            self.eff_params = None

    def check_params(self):
        # Check by calculating epsilons
        is_ok, penalty, epsl, epsr = cubicvol_check_params(self.params)
        return is_ok, penalty

    def dump_params(self):
        data = {'atm': self.params[0], 'skew': self.params[1], 'kurt': self.params[2],
                'vl': self.params[3], 'vr': self.params[4]}
        return data

    def constraints(self):
        # atm, skew, kurt, vl, vr
        lw_bounds = [0.01, 0.0, -0.5, 0.01, 0.01]
        up_bounds = [1.5, 1.0, 3.0, 2.0, 2.0]
        bounds = opt.Bounds(lw_bounds, up_bounds, keep_feasible=False)
        return bounds


def cubicvoleps_formula(t, x, params):
    """ Wrapper on CubicVol formula to take parameter vector as input """
    return cubicvoleps(t, x, *params)


def cubicvol_check_params(params):
    is_ok = False
    epsl = epsr = None
    if len(params) == 5:
        atm = params[0]
        skew = params[1]
        kurt = params[2]
        vl = params[3]
        vr = params[4]

        is_ok = (atm >= 0.0 and skew >= 0.0 and vl >= 0.0 and vr >= 0.0)
        if is_ok:
            try:
                epsl = calculate_epsilon(atm, skew, kurt, vl, False)
                epsr = calculate_epsilon(atm, skew, kurt, vr, True)
            except:
                is_ok = False

    # Sudden death for now
    penalty = 0.0 if is_ok else DEATH_PENALTY
    return is_ok, penalty, epsl, epsr


def cubicvol(t, x, atm, skew, kurt, vl, vr):
    epsl = calculate_epsilon(atm, skew, kurt, vl, False)
    epsr = calculate_epsilon(atm, skew, kurt, vr, True)
    return cubicvoleps(t, x, atm, skew, kurt, epsl, epsr)


def cubicvoleps(t, x, atm, skew, kurt, epsl, epsr):
    timeThreshold = 0.000001
    xThreshold = 0.000001
    # x_ = -x
    t_ = (timeThreshold if np.abs(t) < timeThreshold else t)
    x_ = -x / np.sqrt(t_)
    # # x_ = -np.log(x) / np.sqrt(t_) # Original model assumed input is moneyness, not log-moneyness

    kurt2 = kurt * kurt

    # Left
    threeEpsl = 3.0 * epsl
    deltal = kurt2 + threeEpsl * skew
    soll = 1000000000.0 if deltal < 0 or np.abs(epsl) < xThreshold else (kurt + np.sqrt(deltal)) / threeEpsl
    xprimel = np.minimum(x_, soll)
    epsprimel = -epsl
    voll = np.maximum(atm + xprimel * (skew + xprimel * (kurt + xprimel * epsprimel)), 0.0)

    # Right
    threeEpsr = 3.0 * epsr
    deltar = kurt2 - threeEpsr * skew
    solr = -1000000000.0 if deltar < 0 or np.abs(epsr) < xThreshold else (-kurt - np.sqrt(deltar)) / threeEpsr
    xprimer = np.maximum(x_, solr)
    epsprimer = epsr
    volr = np.maximum(atm + xprimer * (skew + xprimer * (kurt + xprimer * epsprimer)), 0.0)

    return np.where(x_ > 0.0, voll, volr)
    # if x_ > 0:
    #     xPrime = min(x_, soll)
    #     epsilon = -epsl
    # else:
    #     xPrime = max(x_, solr)
    #     epsilon = epsr

    # return max(atm + xPrime * (skew + xPrime * (kurt + xPrime * epsilon)), 0.0)


def calculate_epsilon(atm, skew, kurt, v, is_right):
    if not is_right and not is_valid_left(atm, skew, kurt, v):
        raise RuntimeError("Invalid CubicVol left parameters")

    if is_right and not is_valid_right(atm, skew, kurt, v):
        raise RuntimeError("Invalid CubicVol right parameters")

    skew2 = skew * skew
    skew3 = skew2 * skew
    deltaV = v - atm
    discri = skew2 + 3.0 * kurt * deltaV
    if discri < 0.0:
        raise RuntimeError("Discrimant is negative in CubicVol surface")

    discri = discri**3

    sgn = -1.0 if is_right else 1.0
    if deltaV == 0.0 and not is_right:
        raise RuntimeError("Invalid CubicVol parameters")
    elif deltaV == 0.0 and is_right:
        if skew == 0.0:
            raise RuntimeError("Invalid CubicVol parameters")
        else:
            return kurt * kurt / (4.0 * skew)
    else:
        num = sgn * 2.0 * skew3 + sgn * 9.0 * skew * kurt * deltaV + 2.0 * np.sqrt(discri)
        return num / (27.0 * deltaV * deltaV)


def is_valid_left(atm, skew, kurt, vl):
    result = True
    eps = 0.0000001
    if skew < 0.0 and kurt < 0.0 and vl < 0.0:
        result = False

    if result and vl < atm + eps:
        result = False

    return result


def is_valid_right(atm, skew, kurt, vr):
    result = True
    eps = 0.0000001
    if skew < 0.0 or kurt < 0.0 or vr < 0.0:
        result = False

    if result and kurt != 0.0 and vr < atm - skew * skew / (3.0 * kurt) + eps:
        result = False

    if result and vr == atm and skew == 0.0:
        result = False

    return result


def sample_params(t, vol=0.25):
    """ Guess parameters for display or optimization initial point """
    atm = vol
    skew = 0.1
    kurt = 0.25
    vl = vol + 0.05
    vr = vol + 0.03
    return np.array([atm, skew, kurt, vl, vr])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t = 1.5
    atm = 0.25
    skew = 0.1
    kurt = 0.25
    vl = 0.30
    vr = 0.27

    ms = np.linspace(0.2, 4.0, 100)
    lms = np.log(ms)
    vols = cubicvol(t, lms, atm, skew, kurt, vl, vr)

    plt.plot(ms, vols)
    plt.show()

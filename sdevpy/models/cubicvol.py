import numpy as np
import matplotlib.pyplot as plt
from sdevpy.models.impliedvol import ParamSection


DEATH_PENALTY = 1e6

## TODO
# * Separate calculation of epsilons from the rest
# * Do epsilons only at parameter check to speed up main calculation
# * Vectorize main calculation
# * Prepare sample/initial point and constraints


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
        super().__init__(time, cubicvol_formula)
        self.model = 'CubicVol'

    def check_params(self):
        return biexp_check_params(self.params)

    def dump_params(self):
        data = {'atm': self.params[0], 'skew': self.params[1], 'kurt': self.params[2],
                'vl': self.params[3], 'vr': self.params[4]}
        return data

    def constraints(self):
        # atm, skew, kurt, vl, vr
        lw_bounds = [0.01, 0.01, 0.01, 0.01, 0.01]
        up_bounds = [1.0, 1.0, 1.0, 2.0, 2.0]
        bounds = opt.Bounds(lw_bounds, up_bounds, keep_feasible=False)
        return bounds


def cubicvol(x, *params):
    """ CubicVol formula """
    # Retrieve parameters
    if (len(params) != 5):
        raise RuntimeError(f"Incorrect parameter size in CubicVol: {len(params)}")

    atm = params[0]
    skew = params[1]
    kurt = params[2]
    vl = params[3]
    vr = params[4]

    # Check constraints
    is_ok, _ = cubicvol_check_params(params)
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



def cubicvol_formula(t, x, params):
    """ Wrapper on CubicVol formula to take parameter vector as input """
    return cubicvol(x, *params)




def volatility(t, x, atm, skew, kurt, vL, vR):
    epsL = calculate_epsilon(atm, skew, kurt, vL, False)
    epsR = calculate_epsilon(atm, skew, kurt, vR, True)

    timeThreshold = 0.000001
    xThreshold = 0.000001
    t_ = (timeThreshold if np.abs(t) < timeThreshold else t)

    x_ = -np.log(x) / np.sqrt(t_)
    kurt2 = kurt * kurt
    threeEpsL = 3.0 * epsL
    threeEpsR = 3.0 * epsR
    deltaL = kurt2 + threeEpsL * skew
    deltaR = kurt2 - threeEpsR * skew
    solL = 1000000000.0 if deltaL < 0 or np.abs(epsL) < xThreshold else (kurt + np.sqrt(deltaL)) / threeEpsL
    solR = -1000000000.0 if deltaR < 0 or np.abs(epsR) < xThreshold else (-kurt - np.sqrt(deltaR)) / threeEpsR

    if x_ > 0:
        xPrime = min(x_, solL)
        epsilon = -epsL
    else:
        xPrime = max(x_, solR)
        epsilon = epsR

    return max(atm + xPrime * (skew + xPrime * (kurt + xPrime * epsilon)), 0.0)


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


def is_valid_left(atm, skew, kurt, vL):
    result = True
    eps = 0.0000001
    if skew < 0.0 and kurt < 0.0 and vL < 0.0:
        result = false

    if result and vL < atm + eps:
        result = false

    return result


def is_valid_right(atm, skew, kurt, vR):
    result = True
    eps = 0.0000001
    if skew < 0.0 or kurt < 0.0 or vR < 0.0:
        result = false

    if result and kurt != 0.0 and vR < atm - skew * skew / (3.0 * kurt) + eps:
        result = false

    if result and vR == atm and skew == 0.0:
        result = false

    return result


if __name__ == "__main__":
    print("Hello")

    t = 1.5
    x = 1.2
    atm = 0.25
    vL = 0.30
    vR = 0.27
    skew = 0.10
    kurt = 0.25


    ms = np.linspace(0.2, 4.0, 100)
    vols = []
    for m in ms:
        vols.append(volatility(t, m, atm, skew, kurt, vL, vR))

    plt.plot(ms, vols)
    plt.show()

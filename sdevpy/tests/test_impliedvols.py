import numpy as np
from sdevpy.models import svi
from sdevpy.models.surfaces.tssvi1 import TsSvi1


def test_tssvi1():
    surface = TsSvi1()
    params = [0.20, 0.25, 0.10, 2.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    surface.calibrated_parameters = params
    print(len(params))

    t = np.asarray([0.5, 1.5, 2.5])
    k = np.asarray([90, 100, 110])
    f = np.asarray([95, 105, 115])
    is_call = True
    test = surface.calculate(t, k, is_call, f)
    ref = np.asarray([0.40565045, 0.26784152, 0.22368161])
    assert np.allclose(test, ref, 1e-10)


def test_svi_formula():
    t = 0.5
    params = svi.sample_params(t)
    params[1] = 0.1 # b
    params[2] = -0.30 # rho
    params[4] = 0.2 # sigma

    m = np.asarray([0.5, 1.0, 2.0]) # Moneyness
    log_m = np.log(m) # Log-moneyness

    test = svi.svi_formula(t, log_m, params)
    ref = np.asarray([0.49837104, 0.32015621, 0.40644314])
    assert np.allclose(test, ref, 1e-10)


if __name__ == "__main__":
    test_tssvi1()

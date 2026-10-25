import numpy as np
import numpy.typing as npt


def rebonato(v: npt.ArrayLike) -> npt.ArrayLike:
    """ Parametric funtion in Rebonato's book, convenient to fit yield and volatility curves """
    a0 = 0.5
    ainf = 2.0
    b = -0.01
    tau = 5.0
    return ainf + (b * v + a0 - ainf) * np.exp(-v / tau)


def nelson_siegel(t: npt.ArrayLike, b0: npt.ArrayLike, b1: npt.ArrayLike, b2: npt.ArrayLike,
                  tau: npt.ArrayLike) -> npt.ArrayLike:
    """ Nelson-Siegel zero-rate """
    x = np.maximum(t, 1e-8) / tau
    f = (1.0 - np.exp(-x)) / x
    return b0 + b1 * f + b2 * (f - np.exp(-x))
